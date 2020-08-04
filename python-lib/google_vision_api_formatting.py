# -*- coding: utf-8 -*-

"""
Classes to format the ouput of api_parallelizer for each recipe:
- extract meaningful columns from the API JSON response
- draw bounding boxes
"""

import logging
from typing import AnyStr, Dict, List, Union
from enum import Enum

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm as tqdm_auto
from PIL import Image, UnidentifiedImageError
from google.cloud import vision
import pandas as pd

import dataiku

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)
from dku_io_utils import PATH_COLUMN
from api_parallelizer import DEFAULT_PARALLEL_WORKERS
from plugin_image_utils import (
    save_image_bytes,
    draw_bounding_box_pil_image,
    draw_bounding_poly_pil_image,
    crop_pil_image,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class UnsafeContentCategoryEnum(Enum):
    ADULT = "Adult"
    SPOOF = "Spoof"
    MEDICAL = "Medical"
    VIOLENCE = "Violence"
    RACY = "Racy"


class TextFeatureType(Enum):
    PAGE = "Page"
    BLOCK = "Block"
    PARAGRAPH = "Paragraph"
    WORD = "Word"
    SYMBOL = "Symbol"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Generic Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply 'format_row' function to dataframe
    - use 'format_image' function on all images and save them to folder
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        self.input_df = input_df
        self.input_folder = input_folder
        self.output_df = None  # initialization before calling format_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.parallel_workers = parallel_workers
        self.api_column_names = build_unique_column_names(input_df.keys(), column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k] for k, v in self.api_column_names._asdict().items()
        }
        self.column_description_dict[PATH_COLUMN] = "Path of the file relative to the input folder"

    def format_row(self, row: Dict) -> Dict:
        """
        Identity function, to be overriden by a real row formatting function
        """
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generic method to apply the format_row method to a dataframe and move API columns at the end
        Do not override this method!
        """
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        self.output_df = df
        return df

    def format_image(self, image: Image, response: Dict) -> Image:
        """
        Identity function, to be overriden by a real image formatting function
        """
        return image

    def format_save_image(self, output_folder: dataiku.Folder, image_path: AnyStr, response: Dict) -> bool:
        """
        Generic method to apply the format_image method to an image in the input_folder using the API response
        and save the formatted image to an output folder.
        Do not override this method!
        """
        result = False
        with self.input_folder.get_download_stream(image_path) as stream:
            try:
                pil_image = Image.open(stream)
                if len(response) != 0:
                    formatted_image = self.format_image(pil_image, response)
                else:
                    formatted_image = pil_image.copy()
                image_bytes = save_image_bytes(formatted_image, image_path)
                output_folder.upload_stream(image_path, image_bytes.getvalue())
                result = True
            except (UnidentifiedImageError, ValueError, TypeError, OSError) as e:
                logging.warning("Could not annotate image on path: {} because of error: {}".format(image_path, e))
                if self.error_handling == ErrorHandlingEnum.FAIL:
                    logging.exception(e)
        return result

    def format_save_images(self, output_folder: dataiku.Folder):
        """
        Generic method to apply the format_save_image on all images using the output dataframe with API responses
        Do not override this method!
        """
        df_iterator = (i[1].to_dict() for i in self.output_df.iterrows())
        len_iterator = len(self.output_df.index)
        logging.info("Formatting and saving images to output folder...")
        api_results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.format_save_image,
                    output_folder=output_folder,
                    image_path=row[PATH_COLUMN],
                    response=safe_json_loads(row[self.api_column_names.response]),
                )
                for row in df_iterator
            ]
            for f in tqdm_auto(as_completed(futures), total=len_iterator):
                api_results.append(f.result())
        num_success = sum(api_results)
        num_error = len(api_results) - num_success
        logging.info(
            "Formatting and saving images to output folder: {} images succeeded, {} failed".format(
                num_success, num_error
            )
        )


class ContentDetectionLabelingAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Content Detection & Labeling API responses:
    - make sure response is valid JSON
    - extract content labels in a dataset
    - compute column descriptions
    - draw bounding boxes around objects with text containing label name and confidence score
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        content_categories: List[vision.enums.Feature.Type],
        input_folder: dataiku.Folder = None,
        minimum_score: float = 0,
        max_results: int = 10,
        column_prefix: AnyStr = "content_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        super().__init__(
            input_df=input_df,
            input_folder=input_folder,
            column_prefix=column_prefix,
            error_handling=error_handling,
            parallel_workers=parallel_workers,
        )
        self.content_categories = content_categories
        self.minimum_score = float(minimum_score)
        self.max_results = int(max_results)
        self._compute_column_description()

    def _compute_column_description(self):
        """
        Private method to compute output column names and descriptions for the format_row method
        """
        if vision.enums.Feature.Type.LABEL_DETECTION in self.content_categories:
            self.label_list_column = generate_unique("label_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.label_list_column] = "List of labels from the API"
        if vision.enums.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            self.object_list_column = generate_unique("object_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.object_list_column] = "List of objects from the API"
        if vision.enums.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            self.landmark_list_column = generate_unique("landmark_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.landmark_list_column] = "List of landmarks from the API"
        if vision.enums.Feature.Type.LOGO_DETECTION in self.content_categories:
            self.logo_list_column = generate_unique("logo_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.logo_list_column] = "List of logos from the API"
        if vision.enums.Feature.Type.WEB_DETECTION in self.content_categories:
            self.web_label_column = generate_unique("web_label", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.web_label_column] = "Web label from the API"
            self.web_entity_list_column = generate_unique("web_entity_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.web_entity_list_column] = "List of Web entities from the API"
            self.web_full_matching_image_list_column = generate_unique(
                "web_full_matching_image_list", self.input_df.keys(), self.column_prefix
            )
            self.column_description_dict[
                self.web_full_matching_image_list_column
            ] = "List of Web images fully matching the input image"
            self.web_partial_matching_image_list_column = generate_unique(
                "web_partial_matching_image_list", self.input_df.keys(), self.column_prefix
            )
            self.column_description_dict[
                self.web_partial_matching_image_list_column
            ] = "List of Web images partially matching the input image"
            self.web_page_match_list_column = generate_unique(
                "web_page_match_list", self.input_df.keys(), self.column_prefix
            )
            self.column_description_dict[
                self.web_page_match_list_column
            ] = "List of Web pages with images matching the input image"
            self.web_similar_image_list_column = generate_unique(
                "web_similar_image_list", self.input_df.keys(), self.column_prefix
            )
            self.column_description_dict[
                self.web_similar_image_list_column
            ] = "List of Web images visually similar to the input image"

    def _extract_content_list_from_response(
        self,
        response: Dict,
        category_key: AnyStr,
        name_key: AnyStr,
        score_key: AnyStr = None,
        subcategory_key: AnyStr = None,
    ) -> Union[AnyStr, List[AnyStr]]:
        """
        Private method to extract content lists (within a category) from an API response
        """
        formatted_content_list = ""
        if subcategory_key is None:
            content_list = response.get(category_key, [])
        else:
            content_list = response.get(category_key, {}).get(subcategory_key, [])
        if score_key is not None:
            content_list = sorted(
                [l for l in content_list if float(l.get(score_key, 0)) >= self.minimum_score],
                key=lambda x: float(x.get(score_key, 0)),
                reverse=True,
            )
        if len(content_list) != 0:
            formatted_content_list = [l.get(name_key) for l in content_list if l.get(name_key)][: self.max_results]
        return formatted_content_list

    def format_row(self, row: Dict) -> Dict:
        """
        Extracts content lists by category from a row with an API response and assigns them to new columns
        """
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        if vision.enums.Feature.Type.LABEL_DETECTION in self.content_categories:
            row[self.label_list_column] = self._extract_content_list_from_response(
                response, "labelAnnotations", name_key="description", score_key="score"
            )
        if vision.enums.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            row[self.object_list_column] = self._extract_content_list_from_response(
                response, "localizedObjectAnnotations", name_key="name", score_key="score"
            )
        if vision.enums.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            row[self.landmark_list_column] = self._extract_content_list_from_response(
                response, "landmarkAnnotations", name_key="description", score_key="score"
            )
        if vision.enums.Feature.Type.LOGO_DETECTION in self.content_categories:
            row[self.logo_list_column] = self._extract_content_list_from_response(
                response, "logoAnnotations", name_key="description", score_key="score"
            )
        if vision.enums.Feature.Type.WEB_DETECTION in self.content_categories:
            row[self.web_label_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="bestGuessLabels", name_key="label"
            )
            if len(row[self.web_label_column]) != 0:
                row[self.web_label_column] = row[self.web_label_column][0]
            row[self.web_entity_list_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="webEntities", name_key="description", score_key="score",
            )
            row[self.web_full_matching_image_list_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="fullMatchingImages", name_key="url"
            )
            if len(row[self.web_full_matching_image_list_column]) != 0:
                row[self.web_full_matching_image_list_column] = [
                    l for l in row[self.web_full_matching_image_list_column] if "x-raw-image:///" not in l
                ]
            row[self.web_partial_matching_image_list_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="partialMatchingImages", name_key="url"
            )
            row[self.web_page_match_list_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="pagesWithMatchingImages", name_key="url"
            )
            row[self.web_similar_image_list_column] = self._extract_content_list_from_response(
                response, "webDetection", subcategory_key="visuallySimilarImages", name_key="url"
            )
            if len(row[self.web_similar_image_list_column]) != 0:
                row[self.web_similar_image_list_column] = [
                    l for l in row[self.web_similar_image_list_column] if "x-raw-image:///" not in l
                ]
        return row

    def _draw_bounding_box_from_response(
        self, image: Image, response: Dict, category_key: AnyStr, name_key: AnyStr, score_key: AnyStr, color: AnyStr
    ) -> Image:
        """
        Private method to draw bounding boxes on an image from a generic API response
        Expects information on the response keys related to a given content category
        """
        object_annotations = response.get(category_key, [])
        bounding_box_list_dict = sorted(
            [r for r in object_annotations if float(r.get(score_key, 0)) >= self.minimum_score],
            key=lambda x: float(x.get(score_key, 0)),
            reverse=True,
        )[: self.max_results]
        for bounding_box_dict in bounding_box_list_dict:
            bbox_text = "{} - {:.1%} ".format(bounding_box_dict.get(name_key, ""), bounding_box_dict.get(score_key, ""))
            bounding_polygon = bounding_box_dict.get("boundingPoly", {})
            bbox_vertices = []
            use_normalized_coordinates = False
            if "vertices" in bounding_polygon.keys():
                bbox_vertices = bounding_polygon.get("vertices", [])
            if "normalizedVertices" in bounding_polygon.keys():
                bbox_vertices = bounding_polygon.get("normalizedVertices", [])
                use_normalized_coordinates = True
            x_coordinates = [float(v.get("x", 0)) for v in bbox_vertices]
            y_coordinates = [float(v.get("y", 0)) for v in bbox_vertices]
            (ymin, xmin, ymax, xmax) = (min(y_coordinates), min(x_coordinates), max(y_coordinates), max(x_coordinates))
            draw_bounding_box_pil_image(image, ymin, xmin, ymax, xmax, bbox_text, use_normalized_coordinates, color)
        return image

    def format_image(self, image: Image, response: Dict) -> Image:
        """
        Formats images, drawing bounding boxes for all selected content categories
        """
        if vision.enums.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "localizedObjectAnnotations", name_key="name", score_key="score", color="red"
            )
        if vision.enums.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "landmarkAnnotations", name_key="description", score_key="score", color="green"
            )
        if vision.enums.Feature.Type.LOGO_DETECTION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "logoAnnotations", name_key="description", score_key="score", color="blue"
            )
        return image


class UnsafeContentAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Unsafe Content API responses:
    - make sure response is valid JSON
    - extract moderation labels in a dataset
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "moderation_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        unsafe_content_categories: List[UnsafeContentCategoryEnum] = [],
    ):
        super().__init__(
            input_df=input_df,
            input_folder=input_folder,
            column_prefix=column_prefix,
            error_handling=error_handling,
            parallel_workers=parallel_workers,
        )
        self.unsafe_content_categories = unsafe_content_categories
        self._compute_column_description()

    def _compute_column_description(self):
        for n, m in UnsafeContentCategoryEnum.__members__.items():
            category_column = generate_unique(n.lower() + "_likelihood", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[
                category_column
            ] = "Likelihood of category '{}' from VERY_UNLIKELY to VERY_LIKELY".format(m.value)

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        moderation_labels = response.get("safeSearchAnnotation", {})
        for category in self.unsafe_content_categories:
            category_column = generate_unique(
                category.name.lower() + "_likelihood", self.input_df.keys(), self.column_prefix
            )
            row[category_column] = moderation_labels.get(category.name.lower(), "").replace("UNKNOWN", "")
        return row


class CropHintstAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for crop hints API responses:
    - make sure response is valid JSON
    - save cropped images to the folder
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "crop_hints_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        minimum_score: float = 0,
    ):
        super().__init__(
            input_df=input_df,
            input_folder=input_folder,
            column_prefix=column_prefix,
            error_handling=error_handling,
            parallel_workers=parallel_workers,
        )
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        self.score_column = generate_unique("score", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.score_column] = "Confidence score in the crop hint from 0 to 1"
        self.importance_column = generate_unique("importance_fraction", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[
            self.importance_column
        ] = "Importance of the crop hint with respect to the original image from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        crop_hints = response.get("cropHintsAnnotation", {}).get("cropHints", [])
        row[self.score_column] = None
        row[self.importance_column] = None
        if len(crop_hints) != 0:
            row[self.score_column] = crop_hints[0].get("confidence")
            row[self.importance_column] = crop_hints[0].get("importanceFraction")
        return row

    def format_image(self, image: Image, response: Dict) -> Image:
        """
        Crops the image to the given aspect ratio
        """
        crop_hints = [
            h
            for h in response.get("cropHintsAnnotation", {}).get("cropHints", [])
            if float(h.get("confidence", 0)) >= self.minimum_score
        ]
        if len(crop_hints) != 0:
            bounding_polygon = crop_hints[0].get("boundingPoly", {})
            use_normalized_coordinates = False
            if "vertices" in bounding_polygon.keys():
                bbox_vertices = bounding_polygon.get("vertices", [])
            if "normalizedVertices" in bounding_polygon.keys():
                bbox_vertices = bounding_polygon.get("normalizedVertices", [])
                use_normalized_coordinates = True
            x_coordinates = [float(v.get("x", 0)) for v in bbox_vertices]
            y_coordinates = [float(v.get("y", 0)) for v in bbox_vertices]
            (ymin, xmin, ymax, xmax) = (min(y_coordinates), min(x_coordinates), max(y_coordinates), max(x_coordinates))
            image = crop_pil_image(image, ymin, xmin, ymax, xmax, use_normalized_coordinates=use_normalized_coordinates)
        return image


class ImageTextDetectionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Text Detection API responses:
    - make sure response is valid JSON
    - extract list of text transcriptions in a dataset
    - compute column descriptions
    - draw bounding boxes around detected text areas
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "text_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        super().__init__(
            input_df=input_df, input_folder=input_folder, column_prefix=column_prefix, error_handling=error_handling,
        )
        self._compute_column_description()

    def _compute_column_description(self):
        self.text_column_concat = generate_unique("detections_concat", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.text_column_concat] = "Concatenated text detections from the API"
        self.language_code_column = generate_unique("language_code", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.language_code_column] = "Detected language code from the API"
        self.language_score_column = generate_unique("language_score", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[
            self.language_score_column
        ] = "Confidence score in the detected language from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        text_annotations = response.get("fullTextAnnotation", {})
        row[self.text_column_concat] = text_annotations.get("text", "")
        row[self.language_code_column] = ""
        row[self.language_score_column] = None
        pages = text_annotations.get("pages", [])
        if len(pages) != 0:
            detected_languages = sorted(
                pages[0].get("property", {}).get("detectedLanguages", [{}]),
                key=lambda x: float(x.get("confidence", 0)),
                reverse=True,
            )
            if len(detected_languages) != 0:
                row[self.language_code_column] = detected_languages[0].get("languageCode", "")
                row[self.language_score_column] = detected_languages[0].get("confidence")
        return row

    def _get_bounding_polygons(self, response: Dict, feature_type: AnyStr) -> List[Dict]:
        text_annotations = response.get("fullTextAnnotation", {})
        polygons = []
        for page in text_annotations.get("pages", []):
            for block in page.get("blocks", []):
                for paragraph in block.get("paragraphs", []):
                    for word in paragraph.get("words", []):
                        for symbol in word.get("symbols", []):
                            if feature_type == TextFeatureType.SYMBOL:
                                polygons.append(symbol.get("boundingBox", {}))
                        if feature_type == TextFeatureType.WORD:
                            polygons.append(word.get("boundingBox", {}))
                    if feature_type == TextFeatureType.PARAGRAPH:
                        polygons.append(paragraph.get("boundingBox", {}))
                if feature_type == TextFeatureType.BLOCK:
                    polygons.append(block.get("boundingBox", {}))
        return polygons

    def format_image(self, image: Image, response: Dict) -> Image:
        block_polygons = self._get_bounding_polygons(response, TextFeatureType.BLOCK)
        for polygon in block_polygons:
            draw_bounding_poly_pil_image(image, polygon.get("vertices", []), "blue")
        paragraph_polygons = self._get_bounding_polygons(response, TextFeatureType.PARAGRAPH)
        for polygon in paragraph_polygons:
            draw_bounding_poly_pil_image(image, polygon.get("vertices", []), "red")
        word_polygons = self._get_bounding_polygons(response, TextFeatureType.WORD)
        for polygon in word_polygons:
            draw_bounding_poly_pil_image(image, polygon.get("vertices", []), "yellow")
        return image


class DocumentTextDetectionAPIFormatter(ImageTextDetectionAPIFormatter):
    """
    Formatter class for Text Detection API responses:
    - make sure response is valid JSON
    - extract list of text transcriptions in a dataset
    - compute column descriptions
    - draw bounding boxes around detected text areas
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "text_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        super().__init__(
            input_df=input_df, input_folder=input_folder, column_prefix=column_prefix, error_handling=error_handling,
        )

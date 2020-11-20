# -*- coding: utf-8 -*-
"""Module with classes to format Google Cloud Vision API results
- extract meaningful columns from the API JSON response
- draw bounding boxes
"""

import logging
from typing import AnyStr, Dict, List, Union, Tuple
from enum import Enum
from io import BytesIO
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm.auto import tqdm as tqdm_auto
from pdfrw import PdfReader
from pdfrw.errors import PdfError
from google.cloud import vision
from fastcore.utils import store_attr
import pandas as pd

import dataiku

from api_image_formatting import ImageAPIResponseFormatterMeta
from plugin_io_utils import ErrorHandling, generate_unique, safe_json_loads
from image_utils import (
    draw_bounding_box_pil_image,
    draw_bounding_poly_pil_image,
    crop_pil_image,
)
from document_utils import DocumentHandler


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class UnsafeContentCategory(Enum):
    """Enum class to identify each unsafe content category"""

    ADULT = "Adult"
    SPOOF = "Spoof"
    MEDICAL = "Medical"
    VIOLENCE = "Violence"
    RACY = "Racy"


class TextFeatureType(Enum):
    """Enum class to identify each level of detected text"""

    PAGE = "Page"
    BLOCK = "Block"
    PARAGRAPH = "Paragraph"
    WORD = "Word"
    SYMBOL = "Symbol"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class ContentDetectionLabelingAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """Formatter class to format Content Detection & Labeling API results"""

    def __init__(
        self, content_categories: List[vision.Feature.Type], minimum_score: float = 0, max_results: int = 10, **kwargs,
    ):
        store_attr()
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        if vision.Feature.Type.LABEL_DETECTION in self.content_categories:
            self.label_list_column = generate_unique("label_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.label_list_column] = "List of labels from the API"
        if vision.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            self.object_list_column = generate_unique("object_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.object_list_column] = "List of objects from the API"
        if vision.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            self.landmark_list_column = generate_unique("landmark_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.landmark_list_column] = "List of landmarks from the API"
        if vision.Feature.Type.LOGO_DETECTION in self.content_categories:
            self.logo_list_column = generate_unique("logo_list", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[self.logo_list_column] = "List of logos from the API"
        if vision.Feature.Type.WEB_DETECTION in self.content_categories:
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
        """Extract content lists for a given content category from an API response"""
        formatted_content_list = ""
        if not subcategory_key:
            content_list = response.get(category_key, [])
        else:
            content_list = response.get(category_key, {}).get(subcategory_key, [])
        if not score_key:
            content_list = sorted(
                [content for content in content_list if float(content.get(score_key, 0)) >= self.minimum_score],
                key=lambda x: float(x.get(score_key, 0)),
                reverse=True,
            )
        if len(content_list) != 0:
            formatted_content_list = [content.get(name_key) for content in content_list if content.get(name_key)][
                : self.max_results
            ]
        return formatted_content_list

    def format_row(self, row: Dict) -> Dict:
        """Extract content lists for all categories from a row with an API response"""
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        if vision.Feature.Type.LABEL_DETECTION in self.content_categories:
            row[self.label_list_column] = self._extract_content_list_from_response(
                response, "labelAnnotations", name_key="description", score_key="score"
            )
        if vision.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            row[self.object_list_column] = self._extract_content_list_from_response(
                response, "localizedObjectAnnotations", name_key="name", score_key="score"
            )
        if vision.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            row[self.landmark_list_column] = self._extract_content_list_from_response(
                response, "landmarkAnnotations", name_key="description", score_key="score"
            )
        if vision.Feature.Type.LOGO_DETECTION in self.content_categories:
            row[self.logo_list_column] = self._extract_content_list_from_response(
                response, "logoAnnotations", name_key="description", score_key="score"
            )
        if vision.Feature.Type.WEB_DETECTION in self.content_categories:
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
                    match for match in row[self.web_full_matching_image_list_column] if "x-raw-image:///" not in match
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
                    similar for similar in row[self.web_similar_image_list_column] if "x-raw-image:///" not in similar
                ]
        return row

    def _draw_bounding_box_from_response(
        self, image: Image, response: Dict, category_key: AnyStr, name_key: AnyStr, score_key: AnyStr, color: AnyStr
    ) -> Image:
        """Draw content bounding boxes on an image for a given category"""
        object_annotations = response.get(category_key, [])
        bounding_box_list_dict = sorted(
            [
                annotation
                for annotation in object_annotations
                if float(annotation.get(score_key, 0)) >= self.minimum_score
            ],
            key=lambda x: float(x.get(score_key, 0)),
            reverse=True,
        )[: self.max_results]
        for bounding_box_dict in bounding_box_list_dict:
            bbox_text = "{} - {:.1%} ".format(bounding_box_dict.get(name_key, ""), bounding_box_dict.get(score_key, ""))
            bounding_polygon = bounding_box_dict.get("boundingPoly", {})
            bbox_vertices = []
            use_normalized_coordinates = False
            bbox_vertices = bounding_polygon.get("vertices", [])
            normalized_bbox_vertices = bounding_polygon.get("normalizedVertices", [])
            if len(normalized_bbox_vertices) != 0:
                use_normalized_coordinates = True
                bbox_vertices = normalized_bbox_vertices
            if len(bbox_vertices) != 0:
                x_coordinates = [float(vertex.get("x", 0)) for vertex in bbox_vertices]
                y_coordinates = [float(vertex.get("y", 0)) for vertex in bbox_vertices]
                (ymin, xmin, ymax, xmax) = (
                    min(y_coordinates),
                    min(x_coordinates),
                    max(y_coordinates),
                    max(x_coordinates),
                )
                draw_bounding_box_pil_image(image, ymin, xmin, ymax, xmax, bbox_text, use_normalized_coordinates, color)
        return image

    def format_image(self, image: Image, response: Dict) -> Image:
        """Draw content bounding boxes on an image for all categories"""
        if vision.Feature.Type.OBJECT_LOCALIZATION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "localizedObjectAnnotations", name_key="name", score_key="score", color="red"
            )
        if vision.Feature.Type.LANDMARK_DETECTION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "landmarkAnnotations", name_key="description", score_key="score", color="green"
            )
        if vision.Feature.Type.LOGO_DETECTION in self.content_categories:
            image = self._draw_bounding_box_from_response(
                image, response, "logoAnnotations", name_key="description", score_key="score", color="blue"
            )
        return image


class ImageTextDetectionAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """Formatter class to format Image Text Detection API results"""

    def __init__(self, **kwargs):
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        self.text_column_concat = generate_unique("detections_concat", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.text_column_concat] = "Concatenated text detections from the API"
        self.language_code_column = generate_unique("language_code", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.language_code_column] = "Detected language code from the API"
        self.language_score_column = generate_unique("language_score", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[
            self.language_score_column
        ] = "Confidence score in the detected language from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        """Extract detected text and language information from a row with an API response"""
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

    def _get_bounding_polygons(self, response: Dict, feature_type: TextFeatureType) -> List[Dict]:
        """Extract text bounding polygons from an API response for a given text feature"""
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
        """Draw text bounding polygons on an image for text blocks, paragraphs and words"""
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


class DocumentTextDetectionAPIResponseFormatter(ImageTextDetectionAPIResponseFormatter):
    """Formatter class to format Document Text Detection API results

    Inherits from ImageTextDetectionAPIResponseFormatter to reuse methods for response parsing and row formatting

    """

    def __init__(self, **kwargs):
        super()._compute_column_description()
        self.doc_handler = DocumentHandler(
            error_handling=kwargs.get("error_handling"), parallel_workers=kwargs.get("parallel_workers")
        )
        self.column_description_dict[self.doc_handler.PAGE_NUMBER_COLUMN] = "Page number in the document"

    def format_save_tiff_documents(self, output_folder: dataiku.Folder, output_df: pd.DataFrame):
        """Open TIFF documents in a `dataiku.Folder`, draw text bounding polygons and save them to another folder"""
        start = time()
        logging.info(f"Formatting and saving {len(output_df.index)} TIFF pages to output folder...")
        # Reusing existing work done on ImageTextDetectionAPIFormatter for TIFF documents
        (num_success, num_error) = super().format_save_images(
            output_folder=output_folder,
            output_df=output_df,
            path_column=self.doc_handler.SPLITTED_PATH_COLUMN,
            verbose=False,
        )
        logging.info(
            (
                f"Formatting and saving {len(output_df.index)} TIFF pages to output folder: "
                f"{num_success} succeeded, {num_error} failed in {(time() - start):.2f} seconds."
            )
        )
        return (num_success, num_success)

    def format_pdf_document(self, pdf: PdfReader, response: Dict) -> PdfReader:
        """Draw text bounding polygons on a single-page PDF document opened by pdfrw.PdfReader"""
        block_polygons = super()._get_bounding_polygons(response, TextFeatureType.BLOCK)
        for polygon in block_polygons:
            self.doc_handler.draw_bounding_poly_pdf(pdf, polygon.get("normalizedVertices", []), "blue")
        paragraph_polygons = super()._get_bounding_polygons(response, TextFeatureType.PARAGRAPH)
        for polygon in paragraph_polygons:
            self.doc_handler.draw_bounding_poly_pdf(pdf, polygon.get("normalizedVertices", []), "red")
        word_polygons = super()._get_bounding_polygons(response, TextFeatureType.WORD)
        for polygon in word_polygons:
            self.doc_handler.draw_bounding_poly_pdf(pdf, polygon.get("normalizedVertices", []), "yellow")
        return pdf

    def format_save_pdf_document(self, output_folder: dataiku.Folder, pdf_path: AnyStr, response: Dict) -> bool:
        """Open a PDF file in a `dataiku.Folder`, draw text bounding polygons and save it to another folder"""
        result = False
        with self.input_folder.get_download_stream(pdf_path) as stream:
            try:
                pdf = PdfReader(BytesIO(stream.read()))
                if len(response) != 0:
                    pdf = self.format_pdf_document(pdf, response)
                    pdf_bytes = self.doc_handler.save_pdf_bytes(pdf)
                    output_folder.upload_stream(pdf_path, pdf_bytes.getvalue())
                result = True
            except (PdfError, ValueError, TypeError, OSError) as error:
                logging.warning(f"Could not annotate PDF on path: {pdf_path} because of error: {error}")
                if self.error_handling == ErrorHandling.FAIL:
                    logging.exception(error)
        return result

    def format_save_pdf_documents(self, output_folder: dataiku.Folder, output_df: pd.DataFrame) -> Tuple[int, int]:
        """Open PDF documents in a `dataiku.Folder`, draw text bounding polygons and save them to another folder"""
        df_iterator = (index_series_pair[1].to_dict() for index_series_pair in output_df.iterrows())
        len_iterator = len(output_df.index)
        api_results = []
        start = time()
        logging.info(f"Formatting and saving {len_iterator} PDF pages to output folder...")
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.format_save_pdf_document,
                    output_folder=output_folder,
                    pdf_path=row[self.doc_handler.SPLITTED_PATH_COLUMN],
                    response=safe_json_loads(row[self.api_column_names.response]),
                )
                for row in df_iterator
            ]
            for future in tqdm_auto(as_completed(futures), total=len_iterator):
                api_results.append(future.result())
        num_success = sum(api_results)
        num_error = len(api_results) - num_success
        logging.info(
            (
                f"Formatting and saving {len_iterator} PDF pages to output folder: "
                f"{num_success} succeeded, {num_error} failed in {(time() - start):.2f} seconds."
            )
        )
        return (num_success, num_error)

    def format_save_documents(self, output_folder: dataiku.Folder) -> pd.DataFrame:
        """Open PDF/TIFF documents in a `dataiku.Folder`, draw text bounding polygons and save them to another folder"""
        # Split dataframe into PDF and TIFF documents to treat them separately
        output_df_tiff = self.output_df[
            (
                self.output_df[self.doc_handler.SPLITTED_PATH_COLUMN].str.endswith("tif")
                | self.output_df[self.doc_handler.SPLITTED_PATH_COLUMN].str.endswith("tiff")
            )
        ]
        output_df_pdf = self.output_df[(self.output_df[self.doc_handler.SPLITTED_PATH_COLUMN].str.endswith("pdf"))]
        # Format and save TIFF documents
        if len(output_df_tiff.index) != 0:
            (num_success, num_error) = self.format_save_tiff_documents(
                output_folder=output_folder, output_df=output_df_tiff
            )
        # Format and save PDF documents
        if len(output_df_pdf.index) != 0:
            (num_success, num_error) = self.format_save_pdf_documents(
                output_folder=output_folder, output_df=output_df_pdf
            )
        return self.output_df


class UnsafeContentAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """Formatter class to format Unsafe Content Moderation API results"""

    def __init__(self, unsafe_content_categories: List[UnsafeContentCategory] = [], **kwargs):
        store_attr()
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        for name, member in UnsafeContentCategory.__members__.items():
            category_column = generate_unique(name.lower() + "_likelihood", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[
                category_column
            ] = f"Likelihood of category '{member.value}' from 1 (VERY_UNLIKELY) to 5 (VERY_LIKELY)"

    def format_row(self, row: Dict) -> Dict:
        """Extract the likelihood of each unsafe content category from a row with an API response"""
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        moderation_labels = response.get("safeSearchAnnotation", {})
        for category in self.unsafe_content_categories:
            category_column = generate_unique(
                category.name.lower() + "_likelihood", self.input_df.keys(), self.column_prefix
            )
            row[category_column] = moderation_labels.get(category.name.lower(), "")
        return row


class CropHintsAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """Formatter class to format Crop Hints API results"""

    def __init__(self, minimum_score: float = 0, **kwargs):
        store_attr()
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        self.score_column = generate_unique("score", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.score_column] = "Confidence score in the crop hint from 0 to 1"
        self.importance_column = generate_unique("importance_fraction", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[
            self.importance_column
        ] = "Importance of the crop hint with respect to the original image from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        """Extract crop hints annotations from a row with an API response"""
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
        """Crop an image according to the crop hints annotation of an API response"""
        crop_hints = [
            crop_hint
            for crop_hint in response.get("cropHintsAnnotation", {}).get("cropHints", [])
            if float(crop_hint.get("confidence", 0)) >= self.minimum_score
        ]
        if len(crop_hints) != 0:
            bounding_polygon = crop_hints[0].get("boundingPoly", {})
            use_normalized_coordinates = False
            bbox_vertices = bounding_polygon.get("vertices", [])
            normalized_bbox_vertices = bounding_polygon.get("normalizedVertices", [])
            if len(normalized_bbox_vertices) != 0:
                bbox_vertices = normalized_bbox_vertices
                use_normalized_coordinates = True
            if len(bbox_vertices) != 0:
                x_coordinates = [float(vertex.get("x", 0)) for vertex in bbox_vertices]
                y_coordinates = [float(vertex.get("y", 0)) for vertex in bbox_vertices]
                (ymin, xmin, ymax, xmax) = (
                    min(y_coordinates),
                    min(x_coordinates),
                    max(y_coordinates),
                    max(x_coordinates),
                )
                image = crop_pil_image(
                    image, ymin, xmin, ymax, xmax, use_normalized_coordinates=use_normalized_coordinates
                )
        return image

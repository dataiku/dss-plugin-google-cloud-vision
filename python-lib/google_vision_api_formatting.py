# -*- coding: utf-8 -*-
import logging
from typing import AnyStr, Dict, List
from enum import Enum

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm as tqdm_auto
from PIL import Image, UnidentifiedImageError
import pandas as pd

import dataiku

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    IMAGE_PATH_COLUMN,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)
from api_parallelizer import DEFAULT_PARALLEL_WORKERS
from plugin_image_utils import save_image_bytes, auto_rotate_image, draw_bounding_box_pil_image

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class UnsafeContentCategory(Enum):
    EXPLICIT_NUDITY = "Explicit Nudity"
    SUGGESTIVE = "Suggestive"
    VIOLENCE = "Violence"
    VISUALLY_DISTURBING = "Visually Disturbing"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Geric Formatter class for API responses:
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

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        self.output_df = df
        return df

    def format_image(self, image: Image, response: Dict) -> Image:
        return image

    def format_save_image(self, output_folder: dataiku.Folder, image_path: AnyStr, response: Dict) -> bool:
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
            except (UnidentifiedImageError, TypeError, OSError) as e:
                logging.warning("Could not load image on path: " + image_path)
                if self.error_handling == ErrorHandlingEnum.FAIL:
                    raise e
        return result

    def format_save_images(self, output_folder: dataiku.Folder):
        df_iterator = (i[1].to_dict() for i in self.output_df.iterrows())
        len_iterator = len(self.output_df.index)
        logging.info("Saving bounding boxes to output folder...")
        api_results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.format_save_image,
                    output_folder=output_folder,
                    image_path=row[IMAGE_PATH_COLUMN],
                    response=safe_json_loads(row[self.api_column_names.response]),
                )
                for row in df_iterator
            ]
            for f in tqdm_auto(as_completed(futures), total=len_iterator):
                api_results.append(f.result())
        num_success = sum(api_results)
        num_error = len(api_results) - num_success
        logging.info(
            "Saving bounding boxes to output folder: {} images succeeded, {} failed".format(num_success, num_error)
        )


class ObjectDetectionLabelingAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Object Detection & Labeling API responses:
    - make sure response is valid JSON
    - extract object labels in a dataset
    - compute column descriptions
    - draw bounding boxes around objects with text containing label name and confidence score
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        num_objects: int,
        orientation_correction: bool = True,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "object_api",
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
        self.num_objects = int(num_objects)
        self.orientation_correction = bool(orientation_correction)
        self.orientation_column = generate_unique("orientation_correction", input_df.keys(), column_prefix)
        self.label_list_column = generate_unique("label_list", input_df.keys(), column_prefix)
        self.label_name_columns = [
            generate_unique("label_" + str(n + 1) + "_name", input_df.keys(), column_prefix) for n in range(num_objects)
        ]
        self.label_score_columns = [
            generate_unique("label_" + str(n + 1) + "_score", input_df.keys(), column_prefix)
            for n in range(num_objects)
        ]
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.label_list_column] = "List of object labels from the API"
        self.column_description_dict[self.orientation_column] = "Orientation correction detected by the API"
        for n in range(self.num_objects):
            label_column = self.label_name_columns[n]
            score_column = self.label_score_columns[n]
            self.column_description_dict[label_column] = "Object label {} extracted by the API".format(n + 1)
            self.column_description_dict[score_column] = "Confidence score in label {} from 0 to 1".format(n + 1)

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        row[self.label_list_column] = ""
        labels = sorted(response.get("Labels", []), key=lambda x: x.get("Confidence"), reverse=True)
        if len(labels) != 0:
            row[self.label_list_column] = [l.get("Name") for l in labels]
        for n in range(self.num_objects):
            if len(labels) > n:
                row[self.label_name_columns[n]] = labels[n].get("Name", "")
                row[self.label_score_columns[n]] = labels[n].get("Confidence", "")
            else:
                row[self.label_name_columns[n]] = ""
                row[self.label_score_columns[n]] = None
        if self.orientation_correction:
            row[self.orientation_column] = response.get("OrientationCorrection", "")
        return row

    def format_image(self, image: Image, response: Dict) -> Image:
        bounding_box_list_dict = [
            {
                "name": label.get("Name", ""),
                "bbox_dict": instance.get("BoundingBox", {}),
                "confidence": float(instance.get("Confidence") / 100.0),
            }
            for label in response.get("Labels", [])
            for instance in label.get("Instances", [])
        ]
        if self.orientation_correction:
            detected_orientation = response.get("OrientationCorrection", "")
            (image, rotated) = auto_rotate_image(image, detected_orientation)
        bounding_box_list_dict = sorted(bounding_box_list_dict, key=lambda x: x.get("confidence"))
        for bounding_box_dict in bounding_box_list_dict:
            bbox_text = "{} - {:.1%} ".format(bounding_box_dict["name"], bounding_box_dict["confidence"])
            ymin = bounding_box_dict["bbox_dict"].get("Top")
            xmin = bounding_box_dict["bbox_dict"].get("Left")
            ymax = ymin + bounding_box_dict["bbox_dict"].get("Height")
            xmax = xmin + bounding_box_dict["bbox_dict"].get("Width")
            draw_bounding_box_pil_image(image, ymin, xmin, ymax, xmax, bbox_text)
        return image


class TextDetectionAPIFormatter(GenericAPIFormatter):
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
        minimum_score: float = 0,
        orientation_correction: bool = True,
        column_prefix: AnyStr = "text_api",
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
        self.minimum_score = float(minimum_score)
        self.orientation_correction = bool(orientation_correction)
        self.orientation_column = generate_unique("orientation_correction", input_df.keys(), column_prefix)
        self.text_column_list = generate_unique("detections_list", input_df.keys(), column_prefix)
        self.text_column_concat = generate_unique("detections_concat", input_df.keys(), column_prefix)
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.text_column_list] = "List of text detections from the API"
        self.column_description_dict[self.text_column_concat] = "Concatenated text detections from the API"
        self.column_description_dict[self.orientation_column] = "Orientation correction detected by the API"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        text_detections = response.get("TextDetections", [])
        text_detections_filtered = [
            t for t in text_detections if t.get("Confidence") >= self.minimum_score and t.get("ParentId") is None
        ]
        row[self.text_column_list] = ""
        row[self.text_column_concat] = ""
        if len(text_detections_filtered) != 0:
            row[self.text_column_list] = [t.get("DetectedText", "") for t in text_detections_filtered]
            row[self.text_column_concat] = " ".join(row[self.text_column_list])
        if self.orientation_correction:
            row[self.orientation_column] = response.get("OrientationCorrection", "")
        return row

    def format_image(self, image: Image, response: Dict) -> Image:
        text_detections = response.get("TextDetections", [])
        text_bounding_boxes = [
            t.get("Geometry", {}).get("BoundingBox", {})
            for t in text_detections
            if t.get("Confidence") >= self.minimum_score and t.get("ParentId") is None
        ]
        if self.orientation_correction:
            detected_orientation = response.get("OrientationCorrection", "")
            (image, rotated) = auto_rotate_image(image, detected_orientation)
        for bbox in text_bounding_boxes:
            ymin = bbox.get("Top")
            xmin = bbox.get("Left")
            ymax = bbox.get("Top") + bbox.get("Height")
            xmax = bbox.get("Left") + bbox.get("Width")
            draw_bounding_box_pil_image(image=image, ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
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
        category_level: UnsafeContentCategoryLevelEnum = UnsafeContentCategoryLevelEnum.TOP,
        content_categories_top_level: List[UnsafeContentCategoryTopLevelEnum] = [],
        content_categories_second_level: List[UnsafeContentCategorySecondLevelEnum] = [],
        column_prefix: AnyStr = "moderation_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(
            input_df=input_df, column_prefix=column_prefix, error_handling=error_handling,
        )
        self.category_level = category_level
        if self.category_level == UnsafeContentCategoryLevelEnum.TOP:
            self.content_category_enum = UnsafeContentCategoryTopLevelEnum
            self.content_categories = content_categories_top_level
        else:
            self.content_category_enum = UnsafeContentCategorySecondLevelEnum
            self.content_categories = content_categories_second_level
        self.is_unsafe_column = generate_unique("unsafe_content", self.input_df.keys(), self.column_prefix)
        self.unsafe_list_column = generate_unique("unsafe_categories", self.input_df.keys(), self.column_prefix)
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.is_unsafe_column] = "Unsafe content detected by the API"
        self.column_description_dict[self.unsafe_list_column] = "List of unsafe content categories detected by the API"
        for n, m in self.content_category_enum.__members__.items():
            confidence_column = generate_unique(n.lower() + "_score", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[confidence_column] = "Confidence score in category '{}' from 0 to 1".format(
                m.value
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        moderation_labels = response.get("ModerationLabels", [])
        row[self.is_unsafe_column] = False
        row[self.unsafe_list_column] = ""
        unsafe_list = []
        for category in self.content_categories:
            confidence_column = generate_unique(
                category.name.lower() + "_score", self.input_df.keys(), self.column_prefix
            )
            row[confidence_column] = ""
            if self.category_level == UnsafeContentCategoryLevelEnum.TOP:
                scores = [l.get("Confidence") for l in moderation_labels if l.get("ParentName", "") == category.value]
            else:
                scores = [l.get("Confidence") for l in moderation_labels if l.get("Name", "") == category.value]
            if len(scores) != 0:
                unsafe_list.append(str(category.value))
                row[confidence_column] = scores[0]
        if len(unsafe_list) != 0:
            row[self.is_unsafe_column] = True
            row[self.unsafe_list_column] = unsafe_list
        return row

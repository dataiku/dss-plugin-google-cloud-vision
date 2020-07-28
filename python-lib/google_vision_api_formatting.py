# -*- coding: utf-8 -*-

"""
Classes to format the ouput of api_parallelizer for each recipe:
- extract meaningful columns from the API JSON response
- draw bounding boxes
"""

import logging
from typing import AnyStr, Dict  # , List

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm as tqdm_auto
from PIL import Image, UnidentifiedImageError
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
from plugin_image_utils import save_image_bytes, draw_bounding_box_pil_image


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
            except (UnidentifiedImageError, ValueError, TypeError, OSError) as e:
                logging.warning("Could not annotate image on path: {} because of error: {}".format(image_path, e))
                if self.error_handling == ErrorHandlingEnum.FAIL:
                    logging.exception(e)
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
            "Saving bounding boxes to output folder: {} images succeeded, {} failed".format(num_success, num_error)
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
        self.minimum_score = float(minimum_score)
        self.max_results = int(max_results)
        self.label_list_column = generate_unique("label_list", input_df.keys(), column_prefix)
        self.label_name_columns = [
            generate_unique("label_" + str(n + 1) + "_name", input_df.keys(), column_prefix) for n in range(max_results)
        ]
        self.label_score_columns = [
            generate_unique("label_" + str(n + 1) + "_score", input_df.keys(), column_prefix)
            for n in range(max_results)
        ]
        # self._compute_column_description()

    # def _compute_column_description(self):
    #     self.column_description_dict[self.label_list_column] = "List of object labels from the API"
    #     self.column_description_dict[self.orientation_column] = "Orientation correction detected by the API"
    #     for n in range(self.num_objects):
    #         label_column = self.label_name_columns[n]
    #         score_column = self.label_score_columns[n]
    #         self.column_description_dict[label_column] = "Object label {} extracted by the API".format(n + 1)
    #         self.column_description_dict[score_column] = "Confidence score in label {} from 0 to 1".format(n + 1)

    # def format_row(self, row: Dict) -> Dict:
    #     raw_response = row[self.api_column_names.response]
    #     response = safe_json_loads(raw_response, self.error_handling)
    #     row[self.label_list_column] = ""
    #     labels = sorted(response.get("Labels", []), key=lambda x: x.get("Confidence"), reverse=True)
    #     if len(labels) != 0:
    #         row[self.label_list_column] = [l.get("Name") for l in labels]
    #     for n in range(self.num_objects):
    #         if len(labels) > n:
    #             row[self.label_name_columns[n]] = labels[n].get("Name", "")
    #             row[self.label_score_columns[n]] = labels[n].get("Confidence", "")
    #         else:
    #             row[self.label_name_columns[n]] = ""
    #             row[self.label_score_columns[n]] = None
    #     if self.orientation_correction:
    #         row[self.orientation_column] = response.get("OrientationCorrection", "")
    #     return row

    def format_image(self, image: Image, response: Dict) -> Image:
        object_annotations = response.get("localizedObjectAnnotations", [])
        bounding_box_list_dict = sorted(
            [r for r in object_annotations if r.get("score") >= self.minimum_score], key=lambda x: x.get("score")
        )
        for bounding_box_dict in bounding_box_list_dict:
            bbox_text = "{} - {:.1%} ".format(bounding_box_dict.get("name", ""), bounding_box_dict.get("score", ""))
            bbox_vertices = bounding_box_dict.get("boundingPoly", {}).get("normalizedVertices", [])
            bbox_x_coordinates = [float(v.get("x")) for v in bbox_vertices]
            bbox_y_coordinates = [float(v.get("y")) for v in bbox_vertices]
            ymin = min(bbox_y_coordinates)
            xmin = min(bbox_x_coordinates)
            ymax = max(bbox_y_coordinates)
            xmax = max(bbox_x_coordinates)
            draw_bounding_box_pil_image(image, ymin, xmin, ymax, xmax, bbox_text)
        return image

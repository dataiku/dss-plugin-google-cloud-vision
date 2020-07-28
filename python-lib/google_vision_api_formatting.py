# -*- coding: utf-8 -*-

"""
Classes to format the ouput of api_parallelizer for each recipe:
- extract meaningful columns from the API JSON response
- draw bounding boxes
"""

import logging
from typing import AnyStr, Dict  # , List
from enum import Enum

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm as tqdm_auto
from PIL import Image, UnidentifiedImageError
import pandas as pd

import dataiku

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    # generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)
from dku_io_utils import PATH_COLUMN
from api_parallelizer import DEFAULT_PARALLEL_WORKERS
from plugin_image_utils import save_image_bytes  # , draw_bounding_box_pil_image

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

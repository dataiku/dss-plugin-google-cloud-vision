# -*- coding: utf-8 -*-
"""Module with a generic class to format Computer Vision API results"""

import logging
from typing import AnyStr, Dict, Tuple
from time import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm as tqdm_auto
from PIL import Image, UnidentifiedImageError
from fastcore.utils import store_attr
from fastcore.meta import PrePostInitMeta
import pandas as pd

import dataiku

from plugin_io_utils import (
    PATH_COLUMN,
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandling,
    build_unique_column_names,
    safe_json_loads,
    move_api_columns_to_end,
)
from plugin_image_utils import save_image_bytes


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class ComputerVisionAPIFormatter:
    """
    Generic Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply 'format_row' function to dataframe
    - use 'format_image' function on all images and save them to folder
    """

    DEFAULT_PARALLEL_WORKERS = 4

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandling = ErrorHandling.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        **kwargs,
    ):
        store_attr()
        self.output_df = None  # initialization before calling format_df
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
        start = time()
        logging.info(f"Formatting API results with {len(df.index)} rows...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info(f"Formatting API results with {len(df.index)} rows: Done in {(time() - start):.2f}.")
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
                logging.warning(f"Could not annotate image on path: {image_path} because of error: {e}")
                if self.error_handling == ErrorHandling.FAIL:
                    logging.exception(e)
        return result

    def format_save_images(
        self,
        output_folder: dataiku.Folder,
        output_df: pd.DataFrame = None,
        path_column: AnyStr = PATH_COLUMN,
        verbose: bool = True,
    ) -> Tuple[int, int]:
        """
        Generic method to apply the format_save_image on all images using the output dataframe with API responses
        Do not override this method!
        """
        if output_df is None:
            output_df = self.output_df
        df_iterator = (i[1].to_dict() for i in output_df.iterrows())
        len_iterator = len(output_df.index)
        if verbose:
            logging.info(f"Formatting and saving {len_iterator} images to output folder...")
        start = time()
        api_results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.format_save_image,
                    output_folder=output_folder,
                    image_path=row[path_column],
                    response=safe_json_loads(row[self.api_column_names.response]),
                )
                for row in df_iterator
            ]
            for f in tqdm_auto(as_completed(futures), total=len_iterator):
                api_results.append(f.result())
        num_success = sum(api_results)
        num_error = len(api_results) - num_success
        if verbose:
            logging.info(
                (
                    f"Formatting and saving {len_iterator} images to output folder: "
                    f"{num_success} images succeeded, {num_error} failed in {(time() - start):.2f}."
                )
            )
        return (num_success, num_error)


class ComputerVisionAPIFormatterMeta(ComputerVisionAPIFormatter, metaclass=PrePostInitMeta):
    def __pre_init__(self, **kwargs):
        super().__init__(**kwargs)

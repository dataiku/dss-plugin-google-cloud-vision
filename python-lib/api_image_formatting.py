# -*- coding: utf-8 -*-
"""Module with a generic class to format Computer Vision API results"""

import logging
from typing import AnyStr, Dict, Tuple
from time import perf_counter

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
from image_utils import save_image_bytes


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class ImageAPIResponseFormatter:
    """Generic Formatter class to format API results related to images

    This class defines the overall structure which other API Formatter classes should inherit from.

    """

    DEFAULT_PARALLEL_WORKERS = 4
    IMAGE_FORMATTING_EXCEPTIONS = (UnidentifiedImageError, Image.DecompressionBombError, ValueError, TypeError, OSError)

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
            column_name: API_COLUMN_NAMES_DESCRIPTION_DICT[key]
            for key, column_name in self.api_column_names._asdict().items()
        }
        self.column_description_dict[PATH_COLUMN] = "Path of the file relative to the input folder"

    def format_row(self, row: Dict) -> Dict:
        """Identity function to be overriden by a real row formatting function"""
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generic method to apply the format_row method to a dataframe and move API columns at the end

        Do not override this method!

        """
        start = perf_counter()
        logging.info(f"Formatting API results for {len(df.index)} row(s)...")
        self.output_df = df.apply(func=self.format_row, axis=1)
        self.output_df = move_api_columns_to_end(self.output_df, self.api_column_names, self.error_handling)
        logging.info(
            f"Formatting API results for {len(df.index)} row(s): Done in {(perf_counter() - start):.2f} seconds."
        )
        return self.output_df

    def format_image(self, image: Image, response: Dict) -> Image:
        """Identity function to be overriden by a real image formatting function"""
        return image

    def format_save_image(self, output_folder: dataiku.Folder, image_path: AnyStr, response: Dict) -> bool:
        """Generic method to apply `self.format_image` to an image in `self.input_folder` and save it to an `output folder`

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
            except self.IMAGE_FORMATTING_EXCEPTIONS as error:
                logging.warning(f"Could not format image on path: {image_path} because of error: {error}")
                if self.error_handling == ErrorHandling.FAIL:
                    logging.exception(error)
        return result

    def format_save_images(
        self,
        output_folder: dataiku.Folder,
        output_df: pd.DataFrame = None,
        path_column: AnyStr = PATH_COLUMN,
        verbose: bool = True,
    ) -> Tuple[int, int]:
        """Generic method to apply `self.format_save_image` to all images using an `output_df` with API responses

        Do not override this method!

        """
        if output_df is None:
            output_df = self.output_df
        df_iterator = (index_series_pair[1].to_dict() for index_series_pair in output_df.iterrows())
        len_iterator = len(output_df.index)
        if verbose:
            logging.info(f"Formatting and saving {len_iterator} image(s) to output folder...")
        start = perf_counter()
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
            for future in tqdm_auto(as_completed(futures), total=len_iterator):
                api_results.append(future.result())
        num_success = sum(api_results)
        num_error = len(api_results) - num_success
        if verbose:
            logging.info(
                (
                    f"Formatting and saving {len_iterator} image(s) to output folder: "
                    f"{num_success} image(s) succeeded, {num_error} failed in {(perf_counter() - start):.2f} seconds."
                )
            )
        return (num_success, num_error)


class ImageAPIResponseFormatterMeta(ImageAPIResponseFormatter, metaclass=PrePostInitMeta):
    """Meta version of the `ImageAPIFormatter` class to avoid subclassing boilerplate

    See https://fastpages.fast.ai/fastcore/#Avoiding-subclassing-boilerplate for details

    """

    def __pre_init__(self, **kwargs):
        super().__init__(**kwargs)

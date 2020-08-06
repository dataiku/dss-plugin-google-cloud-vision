# -*- coding: utf-8 -*-

"""
Utility functions to manipulate PDF/TIFF documents
Uses the dataiku API for reading/writing
"""

import os
import logging
from typing import AnyStr, List, Dict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
from tqdm.auto import tqdm as tqdm_auto
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.utils import PyPdfError
from PIL import Image, UnidentifiedImageError

import dataiku

from plugin_io_utils import ErrorHandlingEnum
from api_parallelizer import DEFAULT_PARALLEL_WORKERS

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class DocumentSplitError(Exception):
    pass


class DocumentHandler:
    """
    Handles documents (PDF or TIFF)
    - split them into 1 file per page
    - merge splitted files into one document
    """

    INPUT_PATH_KEY = "input_path"
    OUTPUT_PATH_LIST_KEY = "output_path_list"
    SPLITTED_PATH_COLUMN = "splitted_document_path"

    def __init__(
        self,
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        self.error_handling = error_handling
        self.parallel_workers = parallel_workers

    def _split_pdf(
        self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path: AnyStr
    ) -> List[AnyStr]:
        with input_folder.get_download_stream(input_path) as stream:
            input_pdf = PdfFileReader(BytesIO(stream.read()))
        input_path_without_file_name = os.path.split(input_path)[0]
        input_file_name_without_extension = os.path.splitext(os.path.basename(input_path))[0]
        output_path_list = []
        for page in range(input_pdf.getNumPages()):
            pdf_writer = PdfFileWriter()
            pdf_writer.addPage(input_pdf.getPage(page))
            output_path = "{}/{}_page_{}.pdf".format(
                input_path_without_file_name, input_file_name_without_extension, page + 1
            )
            pdf_bytes = BytesIO()
            pdf_writer.write(pdf_bytes)
            output_folder.upload_stream(output_path, pdf_bytes.getvalue())
            output_path_list.append(output_path)
        return output_path_list

    def _split_tiff(
        self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path: AnyStr
    ) -> List[AnyStr]:
        with input_folder.get_download_stream(input_path) as stream:
            pil_image = Image.open(stream)
        input_path_without_file_name = os.path.split(input_path)[0]
        input_file_name_without_extension = os.path.splitext(os.path.basename(input_path))[0]
        page = 0
        output_path_list = []
        while True:
            try:
                pil_image.seek(page)
                output_path = "{}/{}_page_{}.tiff".format(
                    input_path_without_file_name, input_file_name_without_extension, page + 1
                )
                image_bytes = BytesIO()
                pil_image.save(image_bytes, format="TIFF")
                output_folder.upload_stream(output_path, image_bytes.getvalue())
                output_path_list.append(output_path)
                page += 1
            except EOFError:
                break
        return output_path_list

    def split_document(self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path: AnyStr) -> Dict:
        output_dict = {self.INPUT_PATH_KEY: input_path, self.OUTPUT_PATH_LIST_KEY: [""]}
        file_extension = input_path.split(".")[-1].lower()
        try:
            if file_extension == "pdf":
                output_dict[self.OUTPUT_PATH_LIST_KEY] = self._split_pdf(input_folder, output_folder, input_path)
            elif file_extension == "tif" or file_extension == "tiff":
                output_dict[self.OUTPUT_PATH_LIST_KEY] = self._split_tiff(input_folder, output_folder, input_path)
            else:
                raise ValueError("The file does not have the PDF or TIFF extension")
        except (UnidentifiedImageError, PyPdfError, ValueError, TypeError, OSError) as e:
            logging.warning("Could not split document on path: {} because of error: {}".format(input_path, e))
            if self.error_handling == ErrorHandlingEnum.FAIL:
                logging.exception(e)
        return output_dict

    def split_all_documents(
        self, path_df: pd.DataFrame, path_column: AnyStr, input_folder: dataiku.Folder, output_folder: dataiku.Folder
    ) -> pd.DataFrame:
        logging.info("Splitting documents and saving splitted files to output folder...")
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.split_document, input_folder=input_folder, output_folder=output_folder, input_path=input_path
                )
                for input_path in path_df[path_column]
            ]
            for f in tqdm_auto(as_completed(futures), total=len(path_df.index)):
                results.append(f.result())
        num_success = sum([1 if len(d.get(self.OUTPUT_PATH_LIST_KEY, [])) != 0 else 0 for d in results])
        num_error = len(results) - num_success
        logging.info(
            "Splitting documents and saving splitted files to output folder: {} files succeeded, {} failed".format(
                num_success, num_error
            )
        )
        output_df = pd.DataFrame(
            [
                OrderedDict([(path_column, result[self.INPUT_PATH_KEY]), (self.SPLITTED_PATH_COLUMN, output_path)])
                for result in results
                for output_path in result[self.OUTPUT_PATH_LIST_KEY]
            ]
        )
        return output_df

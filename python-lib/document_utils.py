# -*- coding: utf-8 -*-

"""
Utility functions to manipulate PDF/TIFF documents
Uses the dataiku API for reading/writing
"""

import os
import re
import logging
from typing import AnyStr, List, Dict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from time import time

import pandas as pd
from tqdm.auto import tqdm as tqdm_auto
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.utils import PyPdfError
from pdfrw import PdfReader, PdfWriter
from PIL import Image, UnidentifiedImageError
from pdf_annotate import PdfAnnotator, Location, Appearance
from matplotlib import colors
from fastcore.utils import store_attr

import dataiku

from plugin_io_utils import ErrorHandling, PATH_COLUMN

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
    DEFAULT_PARALLEL_WORKERS = 4

    def __init__(
        self, error_handling: ErrorHandling = ErrorHandling.LOG, parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        store_attr()

    def save_pdf_bytes(self, pdf: PdfReader) -> bytes:
        pdf_bytes = BytesIO()
        if len(pdf.pages) != 0:
            pdf_writer = PdfWriter()
            pdf_writer.addpages(pdf.pages)
            pdf_writer.write(pdf_bytes)
        return pdf_bytes

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
            output_path = f"{input_path_without_file_name}/{input_file_name_without_extension}_page_{page + 1}.pdf"
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
                output_path = f"{input_path_without_file_name}/{input_file_name_without_extension}_page_{page+1}.tiff"
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
        file_extension = os.path.splitext(input_path)[1][1:].lower().strip()
        try:
            if file_extension == "pdf":
                output_dict[self.OUTPUT_PATH_LIST_KEY] = self._split_pdf(input_folder, output_folder, input_path)
            elif file_extension == "tif" or file_extension == "tiff":
                output_dict[self.OUTPUT_PATH_LIST_KEY] = self._split_tiff(input_folder, output_folder, input_path)
            else:
                raise ValueError("The file does not have the PDF or TIFF extension")
        except (UnidentifiedImageError, PyPdfError, ValueError, TypeError, OSError) as e:
            logging.warning(f"Cannot split document on path: {input_path} because of error: {e}")
            if self.error_handling == ErrorHandling.FAIL:
                logging.exception(e)
        return output_dict

    def split_all_documents(
        self,
        path_df: pd.DataFrame,
        input_folder: dataiku.Folder,
        output_folder: dataiku.Folder,
        path_column: AnyStr = PATH_COLUMN,
    ) -> pd.DataFrame:
        start = time()
        logging.info(f"Splitting {len(path_df.index)} documents and saving each page to output folder...")
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
        num_success = sum([result[self.OUTPUT_PATH_LIST_KEY][0] != "" for result in results])
        num_error = len(results) - num_success
        num_pages = sum([len(result[self.OUTPUT_PATH_LIST_KEY]) for result in results]) - num_error
        if num_pages == 0:
            raise DocumentSplitError("Could not split any document")
        logging.info(
            (
                f"Splitting {len(path_df.index)} documents and saving each page to output folder: "
                f"{num_success} documents succeeded generating {num_pages} pages, "
                f"{num_error} documents failed in {(time() - start):.2f} seconds."
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

    def _merge_tiff(
        self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path_list: AnyStr, output_path: AnyStr
    ) -> AnyStr:
        # Load all TIFF images in a list
        image_list = []
        for input_path in input_path_list:
            with input_folder.get_download_stream(input_path) as stream:
                image_list.append(Image.open(stream))
        # Save them to a single image object
        image_bytes = BytesIO()
        if len(image_list) > 1:
            image_list[0].save(image_bytes, append_images=image_list[1:], save_all=True, format="TIFF")
        else:
            image_list[0].save(image_bytes, format="TIFF")
        # Save image to output_folder
        output_folder.upload_stream(output_path, image_bytes.getvalue())
        return output_path

    def _merge_pdf(
        self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path_list: AnyStr, output_path: AnyStr
    ) -> AnyStr:
        pdf_writer = PdfFileWriter()
        # Merge all PDF paths in the list
        for path in input_path_list:
            with input_folder.get_download_stream(path) as stream:
                input_pdf = PdfFileReader(BytesIO(stream.read()))
            for page in range(input_pdf.getNumPages()):
                pdf_writer.addPage(input_pdf.getPage(page))
        # Save the merged PDF in the output folder
        pdf_bytes = BytesIO()
        pdf_writer.write(pdf_bytes)
        output_folder.upload_stream(output_path, pdf_bytes.getvalue())
        return output_path

    def extract_page_number_from_path(self, path: AnyStr):
        page_number = ""
        if path:
            pages_found = re.findall(r"page_(\d+)", path)
            if len(pages_found) != 0:
                page_number = int(pages_found[-1])
        return page_number

    def merge_document(
        self, input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path_list: AnyStr, output_path: AnyStr
    ) -> AnyStr:
        if len(input_path_list) == 0:
            raise RuntimeError("No documents to merge")
        file_extension = output_path.split(".")[-1].lower()
        try:
            if input_path_list[0] == "":
                raise ValueError("No files to merge")
            if file_extension == "pdf":
                output_path = self._merge_pdf(input_folder, output_folder, input_path_list, output_path)
                logging.info(f"Merged PDF document: {output_path} with {len(input_path_list)} pages")
            elif file_extension == "tif" or file_extension == "tiff":
                output_path = self._merge_tiff(input_folder, output_folder, input_path_list, output_path)
                logging.info(f"Merged TIFF document: {output_path} with {len(input_path_list)} pages")
            else:
                raise ValueError("No files with PDF/TIFF extension")
            for path in input_path_list:
                input_folder.delete_path(path)
        except (UnidentifiedImageError, PyPdfError, ValueError, TypeError, OSError) as e:
            logging.warning(f"Could not merge document on path: {output_path} because of error: {e}")
            output_path = ""
            if self.error_handling == ErrorHandling.FAIL:
                logging.exception(e)
        return output_path

    def merge_all_documents(
        self, path_df: pd.DataFrame, path_column: AnyStr, input_folder: dataiku.Folder, output_folder: dataiku.Folder
    ):
        output_df_list = path_df.groupby(path_column)[self.SPLITTED_PATH_COLUMN].apply(list).reset_index()
        start = time()
        logging.info(f"Merging {len(path_df.index)} pages of {len(output_df_list.index)} documents...")
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [
                pool.submit(
                    self.merge_document,
                    input_folder=input_folder,
                    output_folder=output_folder,
                    input_path_list=row[1],
                    output_path=row[0],
                )
                for row in output_df_list.itertuples(index=False)
            ]
            for f in tqdm_auto(as_completed(futures), total=len(output_df_list.index)):
                results.append(f.result())
        num_success = sum([1 if output_path != "" else 0 for output_path in results])
        num_error = len(results) - num_success
        logging.info(
            (
                f"Merging {len(path_df.index)} pages of {len(output_df_list.index)} documents..."
                f"{num_success} documents succeeded, {num_error} failed in {(time() - start):.2f} seconds."
            )
        )

    def draw_bounding_poly_pdf(
        self, pdf: PdfReader, vertices: List[Dict], color: AnyStr,
    ):
        """
        Draws a bounding polygon on an pdf, with lines which may not be parallel to the image orientaiton.
        Vertices must be specified in TODO

        Args:
            TODO
        """
        if len(vertices) == 4:
            pdf_annotator = PdfAnnotator(pdf)
            pdf_annotator.set_page_dimensions(dimensions=(1, 1), page_number=0)  # normalize page dimensions
            pdf_annotator.add_annotation(
                annotation_type="polygon",
                location=Location(
                    points=[(vertices[i].get("x", 0.0), 1.0 - vertices[i].get("y", 0.0)) for i in range(4)], page=0,
                ),
                appearance=Appearance(stroke_color=colors.to_rgba(color)),
            )
        else:
            raise ValueError(f"Bounding polygon does not contain 4 vertices: {vertices}")
        return pdf

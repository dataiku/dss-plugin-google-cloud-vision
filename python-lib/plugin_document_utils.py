# -*- coding: utf-8 -*-

"""
Utility functions to manipulate PDF/TIFF documents
Uses the dataiku API for reading/writing
"""
import os
from typing import AnyStr
from io import BytesIO

from PyPDF2 import PdfFileReader, PdfFileWriter
from PIL import Image

import dataiku


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def split_pdf(input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_pdf_path: AnyStr):
    with input_folder.get_download_stream(input_pdf_path) as stream:
        input_pdf = PdfFileReader(BytesIO(stream.read()))
    input_path_without_file_name = os.path.split(input_pdf_path)[0]
    input_file_name_without_extension = os.path.splitext(os.path.basename(input_pdf_path))[0]
    for page in range(input_pdf.getNumPages()):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(input_pdf.getPage(page))
        output_pdf_path = "{}/{}_page_{}.pdf".format(
            input_path_without_file_name, input_file_name_without_extension, page + 1
        )
        pdf_bytes = BytesIO()
        pdf_writer.write(pdf_bytes)
        output_folder.upload_stream(output_pdf_path, pdf_bytes.getvalue())


def split_tiff(input_folder: dataiku.Folder, output_folder: dataiku.Folder, input_path: AnyStr):
    with input_folder.get_download_stream(input_path) as stream:
        pil_image = Image.open(stream)
    input_path_without_file_name = os.path.split(input_path)[0]
    input_file_name_without_extension = os.path.splitext(os.path.basename(input_path))[0]
    page = 0
    while True:
        try:
            pil_image.seek(page)
            output_path = "{}/{}_page_{}.tiff".format(
                input_path_without_file_name, input_file_name_without_extension, page + 1
            )
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format="TIFF")
            output_folder.upload_stream(output_path, image_bytes.getvalue())
            page += 1
        except EOFError:
            break

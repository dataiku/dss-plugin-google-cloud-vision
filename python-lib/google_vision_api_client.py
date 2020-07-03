# -*- coding: utf-8 -*-
import logging
import json
from typing import AnyStr

from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.oauth2.service_account.Credentials import from_service_account_info


class GoogleCloudVisionAPIWrapper:
    """
    Wrapper class for the Google Cloud Vision API client
    """

    API_EXCEPTIONS = (GoogleAPICallError, RetryError)
    SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "gif", "bmp", "webp", "ico"]
    SUPPORTED_DOCUMENT_FORMATS = ["pdf", "tiff", "tif"]

    def __init__(self, gcp_service_account_key: AnyStr = None):
        if gcp_service_account_key is None or gcp_service_account_key == "":
            self.client = vision.ImageAnnotatorClient()
        else:
            credentials = from_service_account_info(json.loads(gcp_service_account_key))
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        logging.info("Credentials loaded")

    def supported_image_format(self, filepath: AnyStr):
        extension = filepath.split(".")[-1].lower()
        return extension in self.SUPPORTED_IMAGE_FORMATS

    def supported_document_format(self, filepath: AnyStr):
        extension = filepath.split(".")[-1].lower()
        return extension in self.SUPPORTED_DOCUMENT_FORMATS

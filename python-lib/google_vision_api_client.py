# -*- coding: utf-8 -*-
import logging
import json
from typing import AnyStr

from google.cloud import language
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.oauth2.service_account.Credentials import from_service_account_info

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (GoogleAPICallError, RetryError)
SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png"]
SUPPORTED_DOCUMENT_FORMATS = ["pdf", "tiff"]

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(gcp_service_account_key: AnyStr = None):
    """
    Get a Google Vision API client from the service account key
    """
    if gcp_service_account_key is None or gcp_service_account_key == "":
        client = language.LanguageServiceClient()
    else:
        credentials = from_service_account_info(json.loads(gcp_service_account_key))
    logging.info("Credentials loaded")
    client = language.LanguageServiceClient(credentials=credentials)
    return client


def supported_image_format(filepath: AnyStr):
    extension = filepath.split(".")[-1].lower()
    return extension in SUPPORTED_IMAGE_FORMATS


def supported_document_format(filepath: AnyStr):
    extension = filepath.split(".")[-1].lower()
    return extension in SUPPORTED_DOCUMENT_FORMATS

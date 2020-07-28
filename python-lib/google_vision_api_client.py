# -*- coding: utf-8 -*-

"""
Wrapper class for the Google Cloud Vision API Python client
Contains utilities functions which are specific to this API
"""

import logging
import json
from typing import AnyStr, List, Dict, NamedTuple, Union

from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from grpc import RpcError
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict


class GoogleCloudVisionAPIWrapper:
    """
    Wrapper class for the Google Cloud Vision API client
    """

    API_EXCEPTIONS = (GoogleAPIError, RpcError)
    SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "gif", "bmp", "webp", "ico"]
    SUPPORTED_DOCUMENT_FORMATS = ["pdf", "tiff", "tif"]

    def __init__(self, gcp_service_account_key: AnyStr = None, gcp_continent: AnyStr = None):
        self.gcp_service_account_key = gcp_service_account_key
        self.gcp_continent = gcp_continent
        self.client_options = None
        if self.gcp_continent is not None and self.gcp_continent != "":
            self.client_options = {"api_endpoint": "{}-vision.googleapis.com".format(self.gcp_continent)}
        self.client = self.get_client()

    def get_client(self) -> vision.ImageAnnotatorClient:
        if self.gcp_service_account_key is None or self.gcp_service_account_key == "":
            client = vision.ImageAnnotatorClient(client_options=self.client_options)
        else:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(self.gcp_service_account_key)
            )
            client = vision.ImageAnnotatorClient(credentials=credentials, client_options=self.client_options)
        logging.info("Credentials loaded")
        return client

    def supported_image_format(self, path: AnyStr) -> bool:
        extension = path.split(".")[-1].lower()
        return extension in self.SUPPORTED_IMAGE_FORMATS

    def supported_document_format(self, path: AnyStr) -> bool:
        extension = path.split(".")[-1].lower()
        return extension in self.SUPPORTED_DOCUMENT_FORMATS

    def batch_api_gcs_image_request(
        self, folder_bucket: AnyStr, folder_root_path: AnyStr, path: AnyStr, **request_kwargs
    ) -> Dict:
        image_uri_dict = {
            "image": {"source": {"image_uri": "gs://{}/{}".format(folder_bucket, folder_root_path + path)}}
        }
        request_dict = {
            **image_uri_dict,
            **request_kwargs,
        }
        return request_dict

    def batch_api_response_parser(
        self, batch: List[Dict], response: Union[Dict, List], api_column_names: NamedTuple
    ) -> Dict:
        """
        Function to parse API results in the batch case. Needed for api_parallelizer.api_call_batch
        when APIs result need specific parsing logic (every API may be different).
        """
        response_dict = MessageToDict(response)
        results = response_dict.get("responses", [])
        for i in range(len(batch)):
            for k in api_column_names:
                batch[i][k] = ""
            error_raw = results[i].get("error", {})
            if len(error_raw) == 0:
                batch[i][api_column_names.response] = json.dumps(results[i])
            batch[i][api_column_names.error_message] = error_raw.get("message", "")
            batch[i][api_column_names.error_type] = error_raw.get("code", "")
            batch[i][api_column_names.error_raw] = error_raw
        return batch

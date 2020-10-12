# -*- coding: utf-8 -*-
"""Module with a wrapper class to call the Google Cloud Vision API"""

import logging
import json
from typing import AnyStr, List, Dict, NamedTuple, Union

from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from grpc import RpcError
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict

import dataiku


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
            self.client_options = {"api_endpoint": f"{self.gcp_continent}-vision.googleapis.com"}
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

    def batch_api_gcs_image_request(
        self, folder_bucket: AnyStr, folder_root_path: AnyStr, path: AnyStr, **request_kwargs
    ) -> Dict:
        image_uri_dict = {"image": {"source": {"image_uri": f"gs://{folder_bucket}/{folder_root_path}{path}"}}}
        request_dict = {
            **image_uri_dict,
            **request_kwargs,
        }
        return request_dict

    def batch_api_gcs_document_request(
        self, folder_bucket: AnyStr, folder_root_path: AnyStr, path: AnyStr, **request_kwargs
    ) -> Dict:
        extension = path.split(".")[-1].lower()
        document_input_config_dict = {
            "input_config": {
                "gcs_source": {"uri": f"gs://{folder_bucket}/{folder_root_path}{path}"},
                "mime_type": "application/pdf" if extension == "pdf" else "image/tiff",
            }
        }
        request_dict = {
            **document_input_config_dict,
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
        results = response_dict.get("responses", [{}])
        if len(results) == 1:
            if "responses" in results[0].keys():
                results = results[0].get("responses", [{}])  # weird edge case with double nesting
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

    def call_api_annotate_image(
        self,
        folder: dataiku.Folder,
        features: Dict,
        image_context: Dict = {},
        row: Dict = None,
        batch: List[Dict] = None,
        path_column: AnyStr = "",
        folder_is_gcs: bool = False,
        folder_bucket: AnyStr = "",
        folder_root_path: AnyStr = "",
    ) -> Union[List[Dict], AnyStr]:
        if folder_is_gcs:
            image_requests = [
                self.batch_api_gcs_image_request(
                    folder_bucket=folder_bucket,
                    folder_root_path=folder_root_path,
                    path=row.get(path_column),
                    features=features,
                    image_context=image_context,
                )
                for row in batch
            ]
            responses = self.client.batch_annotate_images(image_requests)
            return responses
        else:
            image_path = row.get(path_column)
            with folder.get_download_stream(image_path) as stream:
                image_request = {
                    "image": {"content": stream.read()},
                    "features": features,
                    "image_context": image_context,
                }
            response_dict = MessageToDict(self.client.annotate_image(image_request))
            if "error" in response_dict.keys():  # Required as annotate_image does not raise exceptions
                raise GoogleAPIError(response_dict.get("error", {}).get("message", ""))
            return json.dumps(response_dict)

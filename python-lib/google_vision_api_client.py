# -*- coding: utf-8 -*-
"""Module with a wrapper class to call the Google Cloud Vision API"""

import logging
import json
import os
from typing import AnyStr, List, Dict, NamedTuple, Union, Callable

from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from grpc import RpcError
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict
from ratelimit import limits, RateLimitException
from retry import retry
from fastcore.utils import store_attr

import dataiku

from plugin_io_utils import PATH_COLUMN
from document_utils import DocumentHandler, DocumentSplitError


class GoogleCloudVisionAPIWrapper:
    """
    Wrapper class for the Google Cloud Vision API client
    """

    API_EXCEPTIONS = (GoogleAPIError, RpcError)
    SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "gif", "bmp", "webp", "ico"]
    SUPPORTED_DOCUMENT_FORMATS = ["pdf", "tiff", "tif"]
    RATELIMIT_EXCEPTIONS = (RateLimitException, OSError)
    RATELIMIT_RETRIES = 5

    def __init__(
        self,
        gcp_service_account_key: AnyStr = None,
        gcp_continent: AnyStr = None,
        api_quota_period: int = 60,
        api_quota_rate_limit: int = 1800,
    ):
        store_attr()
        self.client = self.get_client()
        self.call_api_annotate_image = self._build_call_api_annotate_image()
        self.call_api_document_text_detection = self._build_call_api_document_text_detection()

    def get_client(self) -> vision.ImageAnnotatorClient:
        client = vision.ImageAnnotatorClient(
            credentials=service_account.Credentials.from_service_account_info(json.loads(self.gcp_service_account_key))
            if self.gcp_service_account_key
            else None,
            client_options={"api_endpoint": f"{self.gcp_continent}-vision.googleapis.com"}
            if self.gcp_continent
            else None,
        )
        logging.info("Credentials loaded")
        return client

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

    def _build_call_api_annotate_image(self) -> Callable:
        @retry(exceptions=self.RATELIMIT_EXCEPTIONS, tries=self.RATELIMIT_RETRIES, delay=self.api_quota_period)
        @limits(calls=self.api_quota_rate_limit, period=self.api_quota_period)
        def call_api_annotate_image(
            folder: dataiku.Folder,
            features: Dict,
            image_context: Dict = {},
            row: Dict = None,
            batch: List[Dict] = None,
            folder_is_gcs: bool = False,
            folder_bucket: AnyStr = "",
            folder_root_path: AnyStr = "",
            **kwargs,
        ) -> Union[List[Dict], AnyStr]:
            image_request = {
                "features": features,
                "image_context": image_context,
            }
            if folder_is_gcs:
                image_requests = [
                    {
                        **{
                            "image": {
                                "source": {"image_uri": f"gs://{folder_bucket}/{folder_root_path}{row[PATH_COLUMN]}"}
                            }
                        },
                        **image_request,
                    }
                    for row in batch
                ]
                responses = self.client.batch_annotate_images(requests=image_requests)
                return responses
            else:
                image_path = row[PATH_COLUMN]
                with folder.get_download_stream(image_path) as stream:
                    image_request["image"] = {"content": stream.read()}
                response_dict = MessageToDict(self.client.annotate_image(request=image_request))
                if "error" in response_dict.keys():  # Required as annotate_image does not raise exceptions
                    raise GoogleAPIError(response_dict.get("error", {}).get("message", ""))
                return json.dumps(response_dict)

        return call_api_annotate_image

    def _build_call_api_document_text_detection(self) -> Callable:
        @retry(exceptions=self.RATELIMIT_EXCEPTIONS, tries=self.RATELIMIT_RETRIES, delay=self.api_quota_period)
        @limits(calls=self.api_quota_rate_limit, period=self.api_quota_period)
        def call_api_document_text_detection(
            folder: dataiku.Folder,
            batch: List[Dict],
            image_context: Dict = {},
            folder_is_gcs: bool = False,
            folder_bucket: AnyStr = "",
            folder_root_path: AnyStr = "",
            **kwargs,
        ) -> List[Dict]:
            document_path = batch[0].get(PATH_COLUMN, "")  # batch contains only 1 page
            splitted_document_path = batch[0].get(DocumentHandler.SPLITTED_PATH_COLUMN, "")
            if splitted_document_path == "":
                raise DocumentSplitError(f"Document could not be split on path: {document_path}")
            extension = os.path.splitext(document_path)[1][1:].lower().strip()
            document_request = {
                "input_config": {"mime_type": "application/pdf" if extension == "pdf" else "image/tiff"},
                "features": [{"type": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION}],
                "image_context": image_context,
            }
            if folder_is_gcs:
                document_request["input_config"]["gcs_source"] = {
                    "uri": f"gs://{folder_bucket}/{folder_root_path}{splitted_document_path}"
                }
            else:
                with folder.get_download_stream(splitted_document_path) as stream:
                    document_request["input_config"]["content"] = stream.read()
            responses = self.client.batch_annotate_files([document_request])
            return responses

        return call_api_document_text_detection

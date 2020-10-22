# -*- coding: utf-8 -*-
"""Module with a wrapper class to call the Google Cloud Vision API"""

import logging
import json
import os
from typing import AnyStr, List, Dict, NamedTuple, Union, Callable
from copy import deepcopy

from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from grpc import RpcError
from proto import Message
from google.oauth2 import service_account
from ratelimit import limits, RateLimitException
from retry import retry
from fastcore.utils import store_attr

import dataiku

from plugin_io_utils import PATH_COLUMN
from document_utils import DocumentHandler, DocumentSplitError


class GoogleCloudVisionAPIWrapper:
    """Wrapper class with helper methods to call the Google Cloud Vision API"""

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
        """Initialize a Google Cloud Vision APIclient"""
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
        self, batch: List[Dict], response: Message, api_column_names: NamedTuple
    ) -> List[Dict]:
        """Parse API results in the Batch case into responses and errors. Used by `api_parallelizer.api_call_batch`."""
        response_dict = json.loads(response.__class__.to_json(response))
        results = response_dict.get("responses", [{}])
        output_batch = deepcopy(batch)
        if len(results) == 1:
            if "responses" in results[0].keys():
                results = results[0].get("responses", [{}])  # weird edge case with double nesting
        for i in range(len(output_batch)):
            for k in api_column_names:
                output_batch[i][k] = ""
            error_raw = results[i].get("error", {})
            if len(error_raw) == 0:
                output_batch[i][api_column_names.response] = json.dumps(results[i])
            else:
                logging.warning(f"Batch API failed on: {batch[i]} because of error: {error_raw.get('message')}")
                output_batch[i][api_column_names.error_message] = error_raw.get("message", "")
                output_batch[i][api_column_names.error_type] = error_raw.get("code", "")
                output_batch[i][api_column_names.error_raw] = error_raw
        return output_batch

    def _build_call_api_annotate_image(self) -> Callable:
        """Build the API calling function for the Image annotation API with retrying and rate limiting"""

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
        ) -> Union[vision.BatchAnnotateImagesResponse, AnyStr]:
            """Call the Google Cloud Vision image annotation API with files stored in a Dataiku managed folder

            Used by `api_parallelizer.api_parallelizer` as `api_call_function` argument
            Activates batching automatically if the Dataiku managed folder is on GCS

            """
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
                response = self.client.annotate_image(request=image_request)
                response_dict = json.loads(response.__class__.to_json(response))
                if "error" in response_dict.keys():  # Required as annotate_image does not raise exceptions
                    raise GoogleAPIError(response_dict.get("error", {}).get("message", ""))
                return json.dumps(response_dict)

        return call_api_annotate_image

    def _build_call_api_document_text_detection(self) -> Callable:
        """Build the API calling function for the document annotation API with retrying and rate limiting"""

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
        ) -> vision.BatchAnnotateFilesResponse:
            """Call the Google Cloud Vision document annotation API with files stored in a Dataiku managed folder

            Used by `api_parallelizer.api_parallelizer` as `api_call_function` argument
            Activates batching automatically if the Dataiku managed folder is on GCS

            """
            document_path = batch[0].get(PATH_COLUMN, "")  # batch contains only 1 page
            splitted_document_path = batch[0].get(DocumentHandler.SPLITTED_PATH_COLUMN, "")
            if splitted_document_path == "":
                raise DocumentSplitError(f"Document could not be split")
            extension = os.path.splitext(document_path)[1][1:].lower().strip()
            document_request = {
                "input_config": {"mime_type": "application/pdf" if extension == "pdf" else "image/tiff"},
                "features": [{"type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION}],
                "image_context": image_context,
            }
            if folder_is_gcs:
                document_request["input_config"]["gcs_source"] = {
                    "uri": f"gs://{folder_bucket}/{folder_root_path}{splitted_document_path}"
                }
            else:
                with folder.get_download_stream(splitted_document_path) as stream:
                    document_request["input_config"]["content"] = stream.read()
            responses = self.client.batch_annotate_files(requests=[document_request])
            return responses

        return call_api_document_text_detection

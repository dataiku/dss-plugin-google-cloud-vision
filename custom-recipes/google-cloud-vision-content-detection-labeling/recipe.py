# -*- coding: utf-8 -*-
import json
from typing import List, Union, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from google.protobuf.json_format import MessageToDict
from google.api_core.exceptions import GoogleAPIError

from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionAPIWrapper
from dku_io_utils import generate_path_df, set_column_description
from plugin_io_utils import PATH_COLUMN
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import ContentDetectionLabelingAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config()
column_prefix = "content_api"

api_wrapper = GoogleCloudVisionAPIWrapper(gcp_service_account_key=config["gcp_service_account_key"])
input_df = generate_path_df(folder=config["input_folder"], path_filter_function=api_wrapper.supported_image_format)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_content_detection(
    max_results: int, row: Dict = None, batch: List[Dict] = None
) -> Union[List[Dict], AnyStr]:
    features = [{"type": c, "max_results": max_results} for c in config["content_categories"]]
    if config["input_folder_is_gcs"]:
        image_requests = [
            api_wrapper.batch_api_gcs_image_request(
                folder_bucket=config["input_folder_bucket"],
                folder_root_path=config["input_folder_root_path"],
                path=row.get(PATH_COLUMN),
                features=features,
            )
            for row in batch
        ]
        responses = api_wrapper.client.batch_annotate_images(image_requests)
        return responses
    else:
        image_path = row.get(PATH_COLUMN)
        with config["input_folder"].get_download_stream(image_path) as stream:
            image_request = {
                "image": {"content": stream.read()},
                "features": features,
            }
        response_dict = MessageToDict(api_wrapper.client.annotate_image(image_request))
        if "error" in response_dict.keys():  # Required as annotate_image does not raise exceptions
            raise GoogleAPIError(response_dict.get("error", {}).get("message", ""))
        return json.dumps(response_dict)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_content_detection,
    api_exceptions=api_wrapper.API_EXCEPTIONS,
    column_prefix=column_prefix,
    parallel_workers=config["parallel_workers"],
    error_handling=config["error_handling"],
    max_results=config["max_results"],
    api_support_batch=config["api_support_batch"],
    batch_size=config["batch_size"],
    batch_api_response_parser=api_wrapper.batch_api_response_parser,
)

api_formatter = ContentDetectionLabelingAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    input_folder=config["input_folder"],
    error_handling=config["error_handling"],
    parallel_workers=config["parallel_workers"],
    content_categories=config["content_categories"],
    minimum_score=config["minimum_score"],
    max_results=config["max_results"],
)
output_df = api_formatter.format_df(df)

config["output_dataset"].write_with_schema(output_df)
set_column_description(
    output_dataset=config["output_dataset"], column_description_dict=api_formatter.column_description_dict
)

if config["output_folder"] is not None:
    api_formatter.format_save_images(config["output_folder"])

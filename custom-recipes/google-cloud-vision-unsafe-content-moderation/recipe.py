# -*- coding: utf-8 -*-
from typing import List, Union, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from google.cloud import vision

from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionAPIWrapper
from dku_io_utils import generate_path_df, set_column_description
from plugin_io_utils import PATH_COLUMN
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import UnsafeContentAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config()
column_prefix = "moderation_api"

api_wrapper = GoogleCloudVisionAPIWrapper(gcp_service_account_key=config["gcp_service_account_key"])
input_df = generate_path_df(folder=config["input_folder"], file_extensions=api_wrapper.SUPPORTED_IMAGE_FORMATS)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_moderation(row: Dict = None, batch: List[Dict] = None) -> Union[List[Dict], AnyStr]:
    results = api_wrapper.call_api_annotate_image(
        row=row,
        batch=batch,
        path_column=PATH_COLUMN,
        folder=config.get("input_folder"),
        folder_is_gcs=config.get("input_folder_is_gcs"),
        folder_bucket=config.get("input_folder_bucket"),
        folder_root_path=config.get("input_folder_root_path"),
        features=[{"type": vision.Feature.Type.SAFE_SEARCH_DETECTION}],
    )
    return results


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_moderation,
    api_exceptions=api_wrapper.API_EXCEPTIONS,
    column_prefix=column_prefix,
    parallel_workers=config["parallel_workers"],
    error_handling=config["error_handling"],
    api_support_batch=config["api_support_batch"],
    batch_size=config["batch_size"],
    batch_api_response_parser=api_wrapper.batch_api_response_parser,
)

api_formatter = UnsafeContentAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    input_folder=config["input_folder"],
    error_handling=config["error_handling"],
    parallel_workers=config["parallel_workers"],
    unsafe_content_categories=config["unsafe_content_categories"],
)
output_df = api_formatter.format_df(df)

config["output_dataset"].write_with_schema(output_df)
set_column_description(
    output_dataset=config["output_dataset"], column_description_dict=api_formatter.column_description_dict
)

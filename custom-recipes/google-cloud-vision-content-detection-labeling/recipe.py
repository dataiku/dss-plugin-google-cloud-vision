# -*- coding: utf-8 -*-
import json
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
import pandas as pd

from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionClient
from google_vision_api_formatting import GenericAPIFormatter
from plugin_io_utils import IMAGE_PATH_COLUMN
from dku_io_utils import generate_path_list, set_column_description
from api_parallelizer import api_parallelizer


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config()
column_prefix = "content_api"
client = GoogleCloudVisionClient(config["api_configuration_preset"])

image_path_list = [
    p for p in generate_path_list(config["input_folder"]) if GoogleCloudVisionClient.supported_image_format(p)
]
input_df = pd.DataFrame(image_path_list, columns=[IMAGE_PATH_COLUMN])
assert len(input_df.index) >= 1


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_content_detection(row: Dict, num_results: int, minimum_score: int) -> AnyStr:
    image_path = row.get(IMAGE_PATH_COLUMN)
    if config["input_folder_is_s3"]:
        image_request = {
            "S3Object": {"Bucket": config["input_folder_bucket"], "Name": config["input_folder_root_path"] + image_path}
        }
    else:
        with config["input_folder"].get_download_stream(image_path) as stream:
            image_request = {"Bytes": stream.read()}
    response = client.detect_moderation_labels(Image=image_request, MinConfidence=minimum_score)
    return json.dumps(response)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_content_detection,
    api_exceptions=GoogleCloudVisionClient.API_EXCEPTIONS,
    parallel_workers=config["parallel_workers"],
    error_handling=config["error_handling"],
    minimum_score=config["minimum_score"],
    api_support_batch=config["api_support_batch"],
)

api_formatter = GenericAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    num_results=config["num_results"],
    input_folder=config["input_folder"],
    error_handling=config["error_handling"],
    parallel_workers=config["parallel_workers"],
)
output_df = api_formatter.format_df(df)

config["output_dataset"].write_with_schema(output_df)
set_column_description(
    output_dataset=config["output_dataset"], column_description_dict=api_formatter.column_description_dict
)

if config["output_folder"] is not None:
    api_formatter.format_save_images(config["output_folder"])

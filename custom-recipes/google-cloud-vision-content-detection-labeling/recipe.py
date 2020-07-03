# -*- coding: utf-8 -*-
import json
from typing import List, Union, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
import pandas as pd
from google.cloud.vision.types import AnnotateImageRequest


from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionAPIWrapper
from google_vision_api_formatting import GenericAPIFormatter
from plugin_io_utils import IMAGE_PATH_COLUMN
from dku_io_utils import generate_image_uri, generate_path_list, set_column_description
from api_parallelizer import api_parallelizer


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config()
column_prefix = "content_api"
client = GoogleCloudVisionAPIWrapper(config["api_configuration_preset"]).client

image_path_list = [
    p for p in generate_path_list(config["input_folder"]) if GoogleCloudVisionAPIWrapper.supported_image_format(p)
]
input_df = pd.DataFrame(image_path_list, columns=[IMAGE_PATH_COLUMN])
assert len(input_df.index) >= 1


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_content_detection(
    num_results: int, minimum_score: int, row: Dict = None, batch: List[Dict] = None,
) -> Union[List[Dict], AnyStr]:
    if config["input_folder_is_gcs"]:
        image_requests = [
            AnnotateImageRequest(
                image=generate_image_uri(
                    config["input_folder_bucket"], config["input_folder_root_path"], row.get(IMAGE_PATH_COLUMN)
                ),
                features=config["content_categories"],
            )
            for row in batch
        ]
        responses = client.batch_annotate_images(image_requests)
        return responses
    else:
        image_path = row.get(IMAGE_PATH_COLUMN)
        with config["input_folder"].get_download_stream(image_path) as stream:
            image_request = AnnotateImageRequest(image=stream.read(), features=config["content_categories"])
        response = client.annotate_image(image_request)
        return json.dumps(response)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_content_detection,
    api_exceptions=GoogleCloudVisionAPIWrapper.API_EXCEPTIONS,
    parallel_workers=config["parallel_workers"],
    error_handling=config["error_handling"],
    minimum_score=config["minimum_score"],
    api_support_batch=config["api_support_batch"],
)

api_formatter = GenericAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    # num_results=config["num_results"],
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

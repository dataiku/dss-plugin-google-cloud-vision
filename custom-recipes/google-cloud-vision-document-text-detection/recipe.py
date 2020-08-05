# -*- coding: utf-8 -*-
from typing import List, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from google.cloud import vision

from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionAPIWrapper
from dku_io_utils import generate_path_df, set_column_description
from plugin_io_utils import PATH_COLUMN
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import DocumentTextDetectionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config()
column_prefix = "text_api"

api_wrapper = GoogleCloudVisionAPIWrapper(gcp_service_account_key=config["gcp_service_account_key"])
input_df = generate_path_df(folder=config["input_folder"], path_filter_function=api_wrapper.supported_document_format)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_text_detection(language_hints: List[AnyStr], batch: List[Dict]) -> List[Dict]:
    # In the particular case of the text detection API for files, only a batch of 1 is allowed
    document_path = batch[0].get(PATH_COLUMN)
    features = [{"type": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION}]
    image_context = {"language_hints": language_hints}
    extension = document_path.split(".")[-1].lower()
    mime_type = "application/pdf" if extension == "pdf" else "image/tiff"
    document_request = {"features": features, "image_context": image_context, "pages": [1]}
    if config["input_folder_is_gcs"]:
        document_request["input_config"] = {
            "gcs_source": {
                "uri": "gs://{}/{}".format(
                    config["input_folder_bucket"], config["input_folder_root_path"] + document_path
                )
            },
            "mime_type": mime_type,
        }
    else:
        with config["input_folder"].get_download_stream(document_path) as stream:
            document_request["input_config"] = ({"content": stream.read(), "mime_type": mime_type},)
    responses = api_wrapper.client.batch_annotate_files([document_request])
    return responses


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_text_detection,
    api_exceptions=api_wrapper.API_EXCEPTIONS,
    column_prefix=column_prefix,
    parallel_workers=config["parallel_workers"],
    error_handling=config["error_handling"],
    api_support_batch=True,  # Need to force this in the specific case of this API
    batch_size=1,  # Need to force this in the specific case of this API
    batch_api_response_parser=api_wrapper.batch_api_response_parser,
    language_hints=config["language_hints"],
)

api_formatter = DocumentTextDetectionAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    input_folder=config["input_folder"],
    error_handling=config["error_handling"],
    parallel_workers=config["parallel_workers"],
)
output_df = api_formatter.format_df(df)

config["output_dataset"].write_with_schema(output_df)
set_column_description(
    output_dataset=config["output_dataset"], column_description_dict=api_formatter.column_description_dict
)

# if config["output_folder"] is not None:
#     api_formatter.format_save_images(config["output_folder"])

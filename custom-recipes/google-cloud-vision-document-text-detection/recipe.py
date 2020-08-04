# -*- coding: utf-8 -*-
import json
import pandas as pd
from typing import List, Union, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from google.cloud import vision
from google.protobuf.json_format import MessageToDict
from google.api_core.exceptions import GoogleAPIError

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
input_df_dict = {
    "images": generate_path_df(config["input_folder"], api_wrapper.supported_image_format),
    "documents": generate_path_df(config["input_folder"], api_wrapper.supported_document_format),
}


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_text_detection_image(
    language_hints: List[AnyStr], row: Dict = None, batch: List[Dict] = None
) -> Union[List[Dict], AnyStr]:
    features = [{"type": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION}]
    image_context = {"language_hints": language_hints}
    if config["input_folder_is_gcs"]:
        image_requests = [
            api_wrapper.batch_api_gcs_image_request(
                folder_bucket=config["input_folder_bucket"],
                folder_root_path=config["input_folder_root_path"],
                path=row.get(PATH_COLUMN),
                features=features,
                image_context=image_context,
            )
            for row in batch
        ]
        responses = api_wrapper.client.batch_annotate_images(image_requests)
        return responses
    else:
        image_path = row.get(PATH_COLUMN)
        with config["input_folder"].get_download_stream(image_path) as stream:
            image_request = {"image": {"content": stream.read()}, "features": features, "image_context": image_context}
        response_dict = MessageToDict(api_wrapper.client.annotate_image(image_request))
        if "error" in response_dict.keys():  # Required as annotate_image does not raise exceptions
            raise GoogleAPIError(response_dict.get("error", {}).get("message", ""))
        return json.dumps(response_dict)


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_text_detection_document(language_hints: List[AnyStr], batch: List[Dict]) -> List[Dict]:
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


call_api_function_dict = {
    "images": call_api_text_detection_image,
    "documents": call_api_text_detection_document,
}
df_dict = {
    k: api_parallelizer(
        input_df=df,
        api_call_function=call_api_function_dict[k],
        api_exceptions=api_wrapper.API_EXCEPTIONS,
        column_prefix=column_prefix,
        parallel_workers=config["parallel_workers"],
        error_handling=config["error_handling"],
        api_support_batch=config["api_support_batch"] if k == "images" else True,  # doc endpoint is always batch
        batch_size=config["batch_size"] if k == "images" else 1,  # doc endpoint limit
        batch_api_response_parser=api_wrapper.batch_api_response_parser,
        language_hints=config["language_hints"],
    )
    for k, df in input_df_dict.items()
}

api_formatter_dict = {
    k: DocumentTextDetectionAPIFormatter(
        input_df=df,
        column_prefix=column_prefix,
        input_folder=config["input_folder"],
        error_handling=config["error_handling"],
        parallel_workers=config["parallel_workers"],
    )
    for k, df in input_df_dict.items()
}

output_df_dict = {k: api_formatter_dict[k].format_df(df) for k, df in df_dict.items()}

config["output_dataset"].write_with_schema(pd.concat(output_df_dict.values()))
set_column_description(
    output_dataset=config["output_dataset"],
    column_description_dict=api_formatter_dict["images"].column_description_dict,
)

if config["output_folder"] is not None:
    api_formatter_dict["images"].format_save_images(config["output_folder"])
    # api_formatter_dict["documents"].format_save_images(config["output_folder"])

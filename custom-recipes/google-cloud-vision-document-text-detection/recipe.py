# -*- coding: utf-8 -*-
from typing import List, Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from google.cloud import vision

from plugin_config_loader import load_plugin_config
from google_vision_api_client import GoogleCloudVisionAPIWrapper
from dku_io_utils import generate_path_df, set_column_description
from plugin_document_utils import DocumentHandler, DocumentSplitError
from plugin_io_utils import PATH_COLUMN
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import DocumentTextDetectionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

config = load_plugin_config(divide_quota_with_batch_size=False)  # edge case
column_prefix = "text_api"

api_wrapper = GoogleCloudVisionAPIWrapper(gcp_service_account_key=config["gcp_service_account_key"])
input_df = generate_path_df(folder=config["input_folder"], path_filter_function=api_wrapper.supported_document_format)
doc_handler = DocumentHandler(error_handling=config["error_handling"], parallel_workers=config["parallel_workers"])
input_df = doc_handler.split_all_documents(
    path_df=input_df,
    path_column=PATH_COLUMN,
    input_folder=config["input_folder"],
    output_folder=config["output_folder"],
)

# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=config["api_quota_period"], tries=5)
@limits(calls=config["api_quota_rate_limit"], period=config["api_quota_period"])
def call_api_text_detection(language_hints: List[AnyStr], batch: List[Dict]) -> List[Dict]:
    # In the particular case of the text detection API for files, only a batch of 1 is allowed
    document_path = batch[0].get(PATH_COLUMN, "")
    splitted_document_path = batch[0].get(doc_handler.SPLITTED_PATH_COLUMN, "")
    if splitted_document_path == "":
        raise DocumentSplitError("Document could not be split on path: {}".format(document_path))
    features = [{"type": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION}]
    image_context = {"language_hints": language_hints}
    extension = document_path.split(".")[-1].lower()
    mime_type = "application/pdf" if extension == "pdf" else "image/tiff"
    document_request = {"input_config": {"mime_type": mime_type}, "features": features, "image_context": image_context}
    if config["output_folder_is_gcs"]:
        document_request["input_config"]["gcs_source"] = {
            "uri": "gs://{}/{}".format(
                config["output_folder_bucket"], config["output_folder_root_path"] + splitted_document_path
            )
        }
    else:
        with config["output_folder"].get_download_stream(splitted_document_path) as stream:
            document_request["input_config"]["content"] = stream.read()
    responses = api_wrapper.client.batch_annotate_files([document_request])
    return responses


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_text_detection,
    api_exceptions=api_wrapper.API_EXCEPTIONS + (DocumentSplitError,),
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
    input_folder=config["output_folder"],  # where splitted documents are located
    error_handling=config["error_handling"],
    parallel_workers=config["parallel_workers"],
)
output_df = api_formatter.format_df(df)

config["output_dataset"].write_with_schema(output_df)
set_column_description(
    output_dataset=config["output_dataset"], column_description_dict=api_formatter.column_description_dict
)


api_formatter.format_save_merge_documents(output_folder=config["output_folder"])

# -*- coding: utf-8 -*-
"""Document Text Detection recipe script"""

from typing import List, Dict
from ratelimit import limits
from retry import retry

from plugin_params_loader import PluginParamsLoader
from plugin_document_utils import DocumentHandler
from plugin_io_utils import PATH_COLUMN
from dku_io_utils import set_column_description
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import DocumentTextDetectionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

params = PluginParamsLoader().validate_load_params()
doc_handler = DocumentHandler(params.error_handling, params.parallel_workers)

# ==============================================================================
# RUN
# ==============================================================================


input_df = doc_handler.split_all_documents(
    path_df=params.input_df,
    path_column=PATH_COLUMN,
    input_folder=params.input_folder,
    output_folder=params.output_folder,
)


@retry(exceptions=params.RATELIMIT_EXCEPTIONS, tries=params.RATELIMIT_RETRIES, delay=params.api_quota_period)
@limits(calls=params.api_quota_rate_limit, period=params.api_quota_period)
def call_api_document_text_detection(batch: List[Dict] = None, **kwargs) -> List[Dict]:
    results = params.api_wrapper.call_api_annotate_image(
        batch=batch,
        folder=params.output_folder,  # where splitted documents are located
        folder_is_gcs=params.output_folder_is_gcs,
        folder_bucket=params.output_folder_bucket,
        folder_root_path=params.output_folder_root_path,
        **kwargs
    )
    return results


df = api_parallelizer(
    api_call_function=call_api_document_text_detection,
    batch_api_response_parser=params.api_wrapper.batch_api_response_parser,
    api_exceptions=params.api_wrapper.API_EXCEPTIONS,
    doc_handler=doc_handler,
    **vars(params)
)

api_formatter = DocumentTextDetectionAPIFormatter(
    input_df=params.input_df,
    column_prefix=params.column_prefix,
    input_folder=params.output_folder,
    error_handling=params.error_handling,
    parallel_workers=params.parallel_workers,
)
output_df = api_formatter.format_df(df)
api_formatter.format_save_merge_documents(output_folder=params.output_folder)

params.output_dataset.write_with_schema(output_df)
set_column_description(params.output_dataset, api_formatter.column_description_dict)

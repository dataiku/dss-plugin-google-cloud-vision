# -*- coding: utf-8 -*-
from typing import List, Union, Dict, AnyStr
from ratelimit import limits
from retry import retry

from plugin_params_loader import PluginParamsLoader
from dku_io_utils import set_column_description
from plugin_io_utils import PATH_COLUMN
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import ContentDetectionLabelingAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

params = PluginParamsLoader().validate_load_params()


# ==============================================================================
# RUN
# ==============================================================================


@retry(exceptions=params.RATELIMIT_EXCEPTIONS, delay=params.api_quota_period, tries=params.NUM_RETRIES)
@limits(calls=params.api_quota_rate_limit, period=params.api_quota_period)
def call_api_content_detection(
    max_results: int, row: Dict = None, batch: List[Dict] = None
) -> Union[List[Dict], AnyStr]:
    results = params.api_wrapper.call_api_annotate_image(
        row=row,
        batch=batch,
        path_column=PATH_COLUMN,
        folder=params.input_folder,
        folder_is_gcs=params.input_folder_is_gcs,
        folder_bucket=params.input_folder_bucket,
        folder_root_path=params.input_folder_root_path,
        features=params.features,
    )
    return results


df = api_parallelizer(
    input_df=params.input_df,
    api_call_function=call_api_content_detection,
    api_exceptions=params.api_wrapper.API_EXCEPTIONS,
    column_prefix=params.column_prefix,
    parallel_workers=params.parallel_workers,
    error_handling=params.error_handling,
    max_results=params.max_results,
    api_support_batch=params.api_support_batch,
    batch_size=params.batch_size,
    batch_api_response_parser=params.api_wrapper.batch_api_response_parser,
)

api_formatter = ContentDetectionLabelingAPIFormatter(
    input_df=params.input_df,
    column_prefix=params.column_prefix,
    input_folder=params.input_folder,
    error_handling=params.error_handling,
    parallel_workers=params.parallel_workers,
    content_categories=params.content_categories,
    minimum_score=params.minimum_score,
    max_results=params.max_results,
)
output_df = api_formatter.format_df(df)

params.output_dataset.write_with_schema(output_df)
set_column_description(
    output_dataset=params.output_dataset, column_description_dict=api_formatter.column_description_dict
)

if params.output_folder:
    api_formatter.format_save_images(params.output_folder)

# -*- coding: utf-8 -*-
"""Image Content Detection & Labeling recipe script"""

from typing import List, Union, Dict, AnyStr
from ratelimit import limits
from retry import retry

from plugin_params_loader import PluginParamsLoader
from dku_io_utils import set_column_description
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import ContentDetectionLabelingAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

params = PluginParamsLoader().validate_load_params()


# ==============================================================================
# RUN
# ==============================================================================


@retry(exceptions=params.RATELIMIT_EXCEPTIONS, tries=params.RATELIMIT_RETRIES, delay=params.api_quota_period)
@limits(calls=params.api_quota_rate_limit, period=params.api_quota_period)
def call_api_annotate_image(row: Dict = None, batch: List[Dict] = None, **kwargs) -> Union[List[Dict], AnyStr]:
    results = params.api_wrapper.call_api_annotate_image(row=row, batch=batch, **kwargs)
    return results


df = api_parallelizer(
    api_call_function=call_api_annotate_image,
    batch_api_response_parser=params.api_wrapper.batch_api_response_parser,
    api_exceptions=params.api_wrapper.API_EXCEPTIONS,
    **vars(params)
)

api_formatter = ContentDetectionLabelingAPIFormatter(**vars(params))
output_df = api_formatter.format_df(df)

params.output_dataset.write_with_schema(output_df)
set_column_description(params.output_dataset, api_formatter.column_description_dict)

if params.output_folder:
    api_formatter.format_save_images(params.output_folder)

# -*- coding: utf-8 -*-
"""Image Unsafe Content Moderation recipe script"""

from plugin_params_loader import PluginParamsLoader, RecipeID
from api_parallelizer import api_parallelizer
from google_vision_api_formatting import UnsafeContentAPIFormatter
from dku_io_utils import set_column_description

params = PluginParamsLoader(RecipeID.UNSAFE_CONTENT_MODERATION).validate_load_params()

df = api_parallelizer(
    api_call_function=params.api_wrapper.call_api_annotate_image,
    batch_api_response_parser=params.api_wrapper.batch_api_response_parser,
    api_exceptions=params.api_wrapper.API_EXCEPTIONS,
    folder=params.input_folder,
    folder_is_gcs=params.input_folder_is_gcs,
    folder_bucket=params.input_folder_bucket,
    folder_root_path=params.input_folder_root_path,
    **vars(params)
)

api_formatter = UnsafeContentAPIFormatter(**vars(params))
output_df = api_formatter.format_df(df)
params.output_dataset.write_with_schema(output_df)
set_column_description(params.output_dataset, api_formatter.column_description_dict)

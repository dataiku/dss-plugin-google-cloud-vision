# -*- coding: utf-8 -*-
"""Document Text Detection recipe script"""

from plugin_params_loader import PluginParamsLoader, RecipeID
from document_utils import DocumentHandler, DocumentSplitError
from parallelizer import parallelizer
from google_vision_api_formatting import DocumentTextDetectionAPIFormatter
from dku_io_utils import set_column_description

params = PluginParamsLoader(RecipeID.DOCUMENT_TEXT_DETECTION).validate_load_params()
doc_handler = DocumentHandler(params.error_handling, params.parallel_workers)

document_df = doc_handler.split_all_documents(
    path_df=params.input_df, input_folder=params.input_folder, output_folder=params.output_folder,
)
params_dict = vars(params)
params_dict.pop("input_df")

df = parallelizer(
    input_df=document_df,
    function=params.api_wrapper.call_api_document_text_detection,
    batch_response_parser=params.api_wrapper.batch_api_response_parser,
    exceptions=params.api_wrapper.API_EXCEPTIONS + (DocumentSplitError,),
    folder=params.output_folder,
    folder_is_gcs=params.output_folder_is_gcs,
    folder_bucket=params.output_folder_bucket,
    folder_root_path=params.output_folder_root_path,
    **params_dict
)

api_formatter = DocumentTextDetectionAPIFormatter(
    input_folder=params.output_folder,  # where splitted documents are stored
    input_df=document_df,
    column_prefix=params.column_prefix,
    error_handling=params.error_handling,
    parallel_workers=params.parallel_workers,
)
api_formatter.format_df(df)
api_formatter.format_save_documents(output_folder=params.output_folder)
output_df = doc_handler.merge_all_documents(
    path_df=api_formatter.output_df, input_folder=params.output_folder, output_folder=params.output_folder,
)
params.output_dataset.write_with_schema(output_df)
set_column_description(params.output_dataset, api_formatter.column_description_dict)

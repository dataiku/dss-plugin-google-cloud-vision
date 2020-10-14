# -*- coding: utf-8 -*-
"""Module with utility classes for validating and loading plugin parameters"""

import logging
import math
from typing import List, Dict, AnyStr

import pandas as pd
from ratelimit import RateLimitException

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from google_vision_api_client import GoogleCloudVisionAPIWrapper
from plugin_io_utils import ErrorHandlingEnum
from dku_io_utils import generate_path_df
from google_vision_api_formatting import UnsafeContentCategoryEnum

# from language_dict import SUPPORTED_LANGUAGES


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


class PluginParams:
    """Class to hold plugin parameters"""

    RATELIMIT_EXCEPTIONS = (RateLimitException, OSError)
    NUM_RETRIES = 5

    def __init__(
        self,
        api_wrapper: GoogleCloudVisionAPIWrapper,
        input_folder: dataiku.Folder,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        input_folder_is_gcs: bool = False,
        input_folder_bucket: AnyStr = "",
        input_folder_root_path: AnyStr = "",
        output_dataset: dataiku.Dataset = None,
        output_folder: dataiku.Folder = None,
        output_folder_is_gcs: bool = False,
        output_folder_bucket: AnyStr = "",
        output_folder_root_path: AnyStr = "",
        api_quota_rate_limit: int = 1800,
        api_quota_period: int = 60,
        api_support_batch: bool = False,
        batch_size: int = 4,
        parallel_workers: int = 4,
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
        annotation_features: List[Dict] = [{}],
        image_context: Dict = {},
        minimum_score: float = 0.0,
        unsafe_content_categories: List[UnsafeContentCategoryEnum] = [],
    ):
        self.api_wrapper = api_wrapper
        self.input_folder = input_folder
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.input_folder_is_gcs = input_folder_is_gcs
        self.input_folder_bucket = input_folder_bucket
        self.input_folder_root_path = input_folder_root_path
        self.output_dataset = output_dataset
        self.output_folder = output_folder
        self.output_folder_is_gcs = output_folder_is_gcs
        self.output_folder_bucket = output_folder_bucket
        self.output_folder_root_path = output_folder_root_path
        self.api_quota_rate_limit = api_quota_rate_limit
        self.api_quota_period = api_quota_period
        self.api_support_batch = api_support_batch
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.error_handling = error_handling
        self.annotation_features = annotation_features
        self.image_context = image_context
        self.minimum_score = minimum_score
        self.unsafe_content_categories = unsafe_content_categories


class PluginParamsLoader:
    """Class to validate and load plugin parameters"""

    RECIPE_ID_COLUMN_PREFIX_MAPPING = {
        "document-text-detection": "text_api",
        "image-text-detection": "text_api",
        "content-detection-labeling": "content_api",
        "unsafe-content-moderation": "moderation_api",
        "cropping": "cropping_api",
    }

    def __init__(self):
        self.recipe_config = get_recipe_config()
        self.recipe_id = self.recipe_config.get("recipe_id")
        self.column_prefix = self.RECIPE_ID_COLUMN_PREFIX_MAPPING[self.recipe_id]
        self.api_support_batch = False  # Changed by `validate_input_params` if input folder is on GCS

    def validate_input_params(self) -> Dict:
        """Validate input parameters"""
        input_params = {}
        input_folder_names = get_input_names_for_role("input_folder")
        if len(input_folder_names) == 0:
            raise PluginParamValidationError("Please specify input folder")
        input_params["input_folder"] = dataiku.Folder(input_folder_names[0])
        if self.recipe_id == "document-text-detection":
            file_extensions = GoogleCloudVisionAPIWrapper.SUPPORTED_DOCUMENT_FORMATS
        else:
            file_extensions = GoogleCloudVisionAPIWrapper.SUPPORTED_IMAGE_FORMATS
        input_params["input_df"] = generate_path_df(
            folder=input_params["input_folder"], file_extensions=file_extensions,
        )
        input_folder_type = input_params["input_folder"].get_info().get("type", "")
        input_params["input_folder_is_gcs"] = input_folder_type == "GCS"
        if input_params["input_folder_is_gcs"]:
            self.api_support_batch = True
            input_folder_access_info = input_params["input_folder"].get_info().get("accessInfo", {})
            input_params["input_folder_bucket"] = input_folder_access_info.get("bucket")
            input_params["input_folder_root_path"] = str(input_folder_access_info.get("root", ""))[1:]
            logging.info("Input folder is stored on GCS, enabling Batch API feature")
        else:
            logging.info(f"Input folder is not stored on GCS ({input_folder_type}), disabling Batch API feature")
        return input_params

    def validate_output_params(self) -> Dict:
        """Validate output parameters"""
        output_params = {}
        # Output dataset
        output_dataset_names = get_output_names_for_role("output_dataset")
        if len(output_dataset_names) == 0:
            raise PluginParamValidationError("Please specify output dataset")
        output_params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
        # Output folder
        output_folder_names = get_output_names_for_role("output_folder")
        output_params["output_folder"] = None
        if len(output_folder_names) == 0 and self.recipe_id != "unsafe-content-moderation":
            raise PluginParamValidationError("Please specify output folder")
        output_params["output_folder"] = dataiku.Folder(output_folder_names[0])
        output_folder_type = output_params["output_folder"].get_info().get("type", "")
        output_params["output_folder_is_gcs"] = output_folder_type == "GCS"
        if output_params["output_folder_is_gcs"]:
            output_folder_access_info = output_params["output_folder"].get_info().get("accessInfo", {})
            output_params["output_folder_bucket"] = output_folder_access_info.get("bucket")
            output_params["output_folder_root_path"] = str(output_folder_access_info.get("root", ""))[1:]
            logging.info("Output folder is stored on GCS")
        else:
            logging.info(f"Output folder is stored on {output_folder_type}")
        return output_params

    def validate_preset_params(self) -> Dict:
        """Validate API configuration preset parameters"""
        preset_params = {}
        api_configuration_preset = self.recipe_config.get("api_configuration_preset", {})
        gcp_continent = api_configuration_preset.get("gcp_continent")
        preset_params["api_wrapper"] = GoogleCloudVisionAPIWrapper(
            gcp_service_account_key=api_configuration_preset.get("gcp_service_account_key"),
            gcp_continent=None if gcp_continent == "auto" else gcp_continent,
        )
        preset_params["api_quota_period"] = int(api_configuration_preset.get("api_quota_period"))
        if preset_params["api_quota_period"] < 1:
            raise PluginParamValidationError("API quota period must be greater than 1")
        preset_params["parallel_workers"] = int(api_configuration_preset.get("parallel_workers"))
        if preset_params["parallel_workers"] < 1 or preset_params["parallel_workers"] > 100:
            raise PluginParamValidationError("Concurrency must be between 1 and 100")
        preset_params["batch_size"] = int(api_configuration_preset.get("batch_size"))
        if preset_params["batch_size"] < 1 or preset_params["batch_size"] > 16:
            raise PluginParamValidationError("Batch size must be between 1 and 16")
        if self.recipe_id == "document-text-detection":
            logging.info("Forcing batch size to 1 in the case of document text detection")
            preset_params["batch_size"] = 1
        preset_params["api_quota_rate_limit"] = int(api_configuration_preset.get("api_quota_rate_limit"))
        if preset_params["api_quota_rate_limit"] < 1:
            raise PluginParamValidationError("API quota rate limit must be greater than 1")
        if self.api_support_batch and self.recipe_id != "document-text-detection":
            preset_params["api_quota_rate_limit"] = max(
                1, math.floor(preset_params["api_quota_rate_limit"] / preset_params["batch_size"])
            )
            logging.info("Dividing API quota rate limit by Batch size")
        preset_params_displayable = {k: v for k, v in preset_params.items() if k != "api_wrapper"}
        logging.info(f"Validated preset parameters: {preset_params_displayable}")
        return preset_params

    def validate_recipe_params(self) -> Dict:
        recipe_params = {}
        return recipe_params

    def validate_load_params(self) -> PluginParams:
        """Validate and load all parameters into a `PluginParams` instance"""
        input_params_dict = self.validate_input_params()
        output_params_dict = self.validate_output_params()
        preset_params_dict = self.validate_preset_params()
        recipe_params_dict = self.validate_recipe_params()
        plugin_params = PluginParams(
            api_support_batch=self.api_support_batch,
            column_prefix=self.column_prefix,
            **input_params_dict,
            **output_params_dict,
            **recipe_params_dict,
            **preset_params_dict,
        )
        return plugin_params

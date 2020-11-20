# -*- coding: utf-8 -*-
"""Module with utility classes for validating and loading plugin parameters"""

import logging
import math
from typing import List, Dict, AnyStr
from enum import Enum

import pandas as pd
from google.cloud import vision
from fastcore.utils import store_attr

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from google_vision_api_client import GoogleCloudVisionAPIWrapper
from google_vision_api_formatting import UnsafeContentCategory
from plugin_io_utils import ErrorHandling
from plugin_io_utils import PATH_COLUMN
from dku_io_utils import generate_path_df

from language_dict import SUPPORTED_LANGUAGES


DOC_URL = "https://www.dataiku.com/product/plugins/google-cloud-vision/"


class RecipeID(Enum):
    """Enum class to identify each recipe"""

    CONTENT_DETECTION_LABELING = "content_api"
    IMAGE_TEXT_DETECTION = "image_text_api"
    DOCUMENT_TEXT_DETECTION = "document_text_api"
    UNSAFE_CONTENT_MODERATION = "moderation_api"
    CROPPING = "cropping_api"


class PluginParamValidationError(ValueError):
    """Custom exception raised when the plugin parameters chosen by the user are invalid"""

    pass


class PluginParams:
    """Class to hold plugin parameters"""

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
        batch_support: bool = False,
        batch_size: int = 4,
        parallel_workers: int = 4,
        error_handling: ErrorHandling = ErrorHandling.LOG,
        features: List[Dict] = [{}],
        max_results: int = 10,
        image_context: Dict = {},
        minimum_score: float = 0.0,
        content_categories: List[vision.Feature.Type] = [],
        unsafe_content_categories: List[UnsafeContentCategory] = [],
        **kwargs,
    ):
        store_attr()


class PluginParamsLoader:
    """Class to validate and load plugin parameters"""

    def __init__(self, recipe_id: RecipeID):
        self.recipe_id = recipe_id
        self.column_prefix = self.recipe_id.value
        self.recipe_config = get_recipe_config()
        self.batch_support = False  # Changed by `validate_input_params` if input folder is on GCS

    def validate_input_params(self) -> Dict:
        """Validate input parameters"""
        input_params = {}
        input_folder_names = get_input_names_for_role("input_folder")
        if len(input_folder_names) == 0:
            raise PluginParamValidationError("Please specify input folder")
        input_params["input_folder"] = dataiku.Folder(input_folder_names[0])
        if self.recipe_id == RecipeID.DOCUMENT_TEXT_DETECTION:
            file_extensions = GoogleCloudVisionAPIWrapper.SUPPORTED_DOCUMENT_FORMATS
            self.batch_support = True
        else:
            file_extensions = GoogleCloudVisionAPIWrapper.SUPPORTED_IMAGE_FORMATS
        input_params["input_df"] = generate_path_df(
            folder=input_params["input_folder"], file_extensions=file_extensions, path_column=PATH_COLUMN
        )
        input_folder_type = input_params["input_folder"].get_info().get("type", "")
        input_params["input_folder_is_gcs"] = input_folder_type == "GCS"
        if input_params["input_folder_is_gcs"]:
            self.batch_support = True
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
        if self.recipe_id != RecipeID.UNSAFE_CONTENT_MODERATION:
            if len(output_folder_names) == 0:
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
        if not api_configuration_preset:
            raise PluginParamValidationError(f"Please specify an API configuration preset according to {DOC_URL}")
        preset_params["gcp_service_account_key"] = api_configuration_preset.get("gcp_service_account_key")
        preset_params["gcp_continent"] = api_configuration_preset.get("gcp_continent")
        if not api_configuration_preset.get("api_quota_period"):
            raise PluginParamValidationError(f"Please specify API quota period in the preset according to {DOC_URL}")
        preset_params["api_quota_period"] = int(api_configuration_preset.get("api_quota_period"))
        if preset_params["api_quota_period"] < 1:
            raise PluginParamValidationError("API quota period must be greater than 1")
        if not api_configuration_preset.get("parallel_workers"):
            raise PluginParamValidationError(f"Please specify concurrency in the preset according to {DOC_URL}")
        preset_params["parallel_workers"] = int(api_configuration_preset.get("parallel_workers"))
        if preset_params["parallel_workers"] < 1 or preset_params["parallel_workers"] > 100:
            raise PluginParamValidationError("Concurrency must be between 1 and 100")
        if not api_configuration_preset.get("batch_size"):
            raise PluginParamValidationError(f"Please specify batch size in the preset according to {DOC_URL}")
        preset_params["batch_size"] = int(api_configuration_preset.get("batch_size"))
        if preset_params["batch_size"] < 1 or preset_params["batch_size"] > 16:
            raise PluginParamValidationError("Batch size must be between 1 and 16")
        if self.recipe_id == RecipeID.DOCUMENT_TEXT_DETECTION:
            logging.info("Forcing batch size to 1 in the case of document text detection")
            preset_params["batch_size"] = 1
        if not api_configuration_preset.get("api_quota_rate_limit"):
            raise PluginParamValidationError(
                f"Please specify API quota rate limit in the preset according to {DOC_URL}"
            )
        preset_params["api_quota_rate_limit"] = int(api_configuration_preset.get("api_quota_rate_limit"))
        if preset_params["api_quota_rate_limit"] < 1:
            raise PluginParamValidationError("API quota rate limit must be greater than 1")
        if self.batch_support:
            preset_params["api_quota_rate_limit"] = max(
                1, math.floor(preset_params["api_quota_rate_limit"] / preset_params["batch_size"])
            )
            logging.info("Dividing API quota rate limit by Batch size")
        preset_params["api_wrapper"] = GoogleCloudVisionAPIWrapper(
            gcp_service_account_key=preset_params["gcp_service_account_key"],
            gcp_continent=None if preset_params["gcp_continent"] == "auto" else preset_params["gcp_continent"],
            api_quota_period=preset_params["api_quota_period"],
            api_quota_rate_limit=preset_params["api_quota_rate_limit"],
        )
        preset_params_displayable = {
            param_name: param_value
            for param_name, param_value in preset_params.items()
            if param_name not in {"gcp_service_account_key", "api_wrapper"}
        }
        logging.info(f"Validated preset parameters: {preset_params_displayable}")
        return preset_params

    def validate_recipe_params(self) -> Dict:
        """Validate recipe parameters"""
        recipe_params = {}
        # Applies to several recipes
        if "minimum_score" in self.recipe_config:
            recipe_params["minimum_score"] = float(self.recipe_config["minimum_score"])
            if recipe_params["minimum_score"] < 0.0 or recipe_params["minimum_score"] > 1.0:
                raise PluginParamValidationError("Minimum score must be between 0 and 1")
        # Applies to content detection & labeling
        if "content_categories" in self.recipe_config:
            recipe_params["content_categories"] = [
                vision.Feature.Type[content_category]
                for content_category in self.recipe_config.get("content_categories", [])
            ]
            if len(recipe_params["content_categories"]) == 0:
                raise PluginParamValidationError("Please select at least one content category")
        if "max_results" in self.recipe_config:
            recipe_params["max_results"] = int(self.recipe_config["max_results"])
            if recipe_params["max_results"] < 1:
                raise PluginParamValidationError("Number of results must be greater than 1")
        # Applies to image and document text detection
        if "language" in self.recipe_config:
            language = self.recipe_config["language"]
            if language not in SUPPORTED_LANGUAGES and language != "":
                raise PluginParamValidationError({f"Invalid language code: {language}"})
            recipe_params["language_hints"] = [language]
        # Applies to document text detection, overrides language if specified
        if "custom_language_hints" in self.recipe_config:
            custom_language_hints = str(self.recipe_config["custom_language_hints"]).replace(" ", "").split(",")
            if len(custom_language_hints) != 0:
                recipe_params["language_hints"] = custom_language_hints
        # Applies to image text detection
        if "text_detection_type" in self.recipe_config:
            recipe_params["text_detection_type"] = vision.Feature.Type[self.recipe_config["text_detection_type"]]
        # Applies to unsafe content moderation
        if "unsafe_content_categories" in self.recipe_config:
            recipe_params["unsafe_content_categories"] = [
                UnsafeContentCategory[category_name]
                for category_name in self.recipe_config.get("unsafe_content_categories", [])
            ]
            if len(recipe_params["unsafe_content_categories"]) == 0:
                raise PluginParamValidationError("Please select at least one unsafe category")
        # Applies to cropping
        if "aspect_ratio" in self.recipe_config:
            recipe_params["aspect_ratio"] = float(self.recipe_config["aspect_ratio"])
            if recipe_params["aspect_ratio"] < 0.1 or recipe_params["aspect_ratio"] > 10:
                raise PluginParamValidationError("Aspect ratio must be between 0.1 and 10")
        logging.info(f"Validated recipe parameters: {recipe_params}")
        return recipe_params

    def validate_load_params(self) -> PluginParams:
        """Validate and load all parameters into a `PluginParams` instance"""
        input_params = self.validate_input_params()
        output_params = self.validate_output_params()
        preset_params = self.validate_preset_params()
        recipe_params = self.validate_recipe_params()
        image_context, features = ({}, {})
        if self.recipe_id == RecipeID.CONTENT_DETECTION_LABELING:
            features = [
                {"type_": content_category, "max_results": recipe_params["max_results"]}
                for content_category in recipe_params["content_categories"]
            ]
        elif self.recipe_id in {RecipeID.IMAGE_TEXT_DETECTION, RecipeID.DOCUMENT_TEXT_DETECTION}:
            image_context = {"language_hints": recipe_params["language_hints"]}
            features = [
                {"type_": recipe_params.get("text_detection_type", vision.Feature.Type.DOCUMENT_TEXT_DETECTION)}
            ]
        elif self.recipe_id == RecipeID.UNSAFE_CONTENT_MODERATION:
            features = [{"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION}]
        elif self.recipe_id == RecipeID.CROPPING:
            image_context = {"crop_hints_params": {"aspect_ratios": [recipe_params["aspect_ratio"]]}}
            features = [{"type_": vision.Feature.Type.CROP_HINTS}]
        plugin_params = PluginParams(
            batch_support=self.batch_support,
            column_prefix=self.column_prefix,
            image_context=image_context,
            features=features,
            **input_params,
            **output_params,
            **recipe_params,
            **preset_params,
        )
        return plugin_params

# -*- coding: utf-8 -*-

"""
Load, resolve and validate the plugin configuration into one clean dictionary
"""

import logging
from typing import Dict, AnyStr
from google.cloud import vision

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from plugin_io_utils import ErrorHandlingEnum
from google_vision_api_formatting import UnsafeContentCategoryEnum
from language_dict import SUPPORTED_LANGUAGES

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def load_plugin_config(mandatory_output: AnyStr = "dataset") -> Dict:
    config = {}
    # Input folder configuration
    input_folder_names = get_input_names_for_role("input_folder")
    config["input_folder"] = dataiku.Folder(input_folder_names[0])
    config["api_support_batch"] = False
    config["input_folder_is_gcs"] = config["input_folder"].get_info().get("type", "") == "GCS"
    if config["input_folder_is_gcs"]:
        logging.info("Input folder is on GCS")
        input_folder_access_info = config["input_folder"].get_info().get("accessInfo", {})
        config["input_folder_bucket"] = input_folder_access_info.get("bucket")
        config["input_folder_root_path"] = str(input_folder_access_info.get("root", ""))[1:]
        config["api_support_batch"] = True
    # Output dataset configuration
    output_dataset_names = get_output_names_for_role("output_dataset")
    config["output_dataset"] = None
    if mandatory_output == "dataset" or len(output_dataset_names) != 0:
        config["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
    # Output folder configuration
    output_folder_names = get_output_names_for_role("output_folder")  # optional output
    config["output_folder"] = None
    if mandatory_output == "folder" or len(output_folder_names) != 0:
        config["output_folder"] = dataiku.Folder(output_folder_names[0])
    # Preset configuration
    recipe_config = get_recipe_config()
    api_configuration_preset = recipe_config.get("api_configuration_preset", {})
    config["gcp_service_account_key"] = str(api_configuration_preset.get("gcp_service_account_key"))
    config["gcp_continent"] = str(api_configuration_preset.get("gcp_continent"))
    if config["gcp_continent"] == "auto":
        config["gcp_continent"] = None
    config["api_quota_rate_limit"] = int(api_configuration_preset.get("api_quota_rate_limit"))
    config["api_quota_period"] = int(api_configuration_preset.get("api_quota_period"))
    config["parallel_workers"] = int(api_configuration_preset.get("parallel_workers"))
    assert config["parallel_workers"] >= 1
    config["batch_size"] = int(api_configuration_preset.get("batch_size"))
    assert config["batch_size"] >= 1
    if config["input_folder_is_gcs"]:
        config["api_quota_rate_limit"] = int(config["api_quota_rate_limit"] / config["batch_size"])
    assert config["api_quota_rate_limit"] >= 1
    # Recipe configuration
    if "content_categories" in recipe_config.keys():
        config["content_categories"] = [
            vision.enums.Feature.Type[c] for c in recipe_config.get("content_categories", [])
        ]
        assert len(config["content_categories"]) >= 1
    if "max_results" in recipe_config.keys():
        config["max_results"] = int(recipe_config.get("max_results", 1))
        assert config["max_results"] >= 1
    if "minimum_score" in recipe_config.keys():
        config["minimum_score"] = float(recipe_config.get("minimum_score", 0))
        assert config["minimum_score"] >= 0.0 and config["minimum_score"] <= 1.0
    if "unsafe_content_categories" in recipe_config.keys():
        config["unsafe_content_categories"] = [
            UnsafeContentCategoryEnum[c] for c in recipe_config["unsafe_content_categories"]
        ]
        assert len(config["unsafe_content_categories"]) >= 1
    if "aspect_ratio" in recipe_config.keys():
        config["aspect_ratio"] = float(recipe_config["aspect_ratio"])
        assert config["aspect_ratio"] >= 0.1 and config["aspect_ratio"] <= 10
    config["error_handling"] = ErrorHandlingEnum[recipe_config.get("error_handling")]
    if "language_hint" in recipe_config.keys():
        config["language_hint"] = recipe_config["language_hint"]
        assert config["language_hint"] in [l.get("value") for l in SUPPORTED_LANGUAGES] + [""]
    return config

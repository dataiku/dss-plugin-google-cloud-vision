# -*- coding: utf-8 -*-
"""Module with read/write utility functions which are *not* based on the Dataiku API"""

import logging
import json
import pandas as pd

from enum import Enum
from typing import AnyStr, List, NamedTuple, Dict
from collections import OrderedDict, namedtuple


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

PATH_COLUMN = "path"
"""Default name of the column to store file paths"""

API_COLUMN_NAMES_DESCRIPTION_DICT = OrderedDict(
    [
        ("response", "Raw response from the API in JSON format"),
        ("error_message", "Error message from the API"),
        ("error_type", "Error type or code from the API"),
        ("error_raw", "Raw error from the API"),
    ]
)
"""Default dictionary of API column names (key) and their descriptions (value)"""


class ErrorHandling(Enum):
    """Enum class to identify how to handle API errors"""

    LOG = "Log"
    FAIL = "Fail"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_unique(name: AnyStr, existing_names: List, prefix: AnyStr) -> AnyStr:
    """Generate a unique name among existing ones by suffixing a number and adding a prefix"""
    if prefix:
        new_name = f"{prefix}_{name}"
    else:
        new_name = name
    for i in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = f"{name}_{i}"
    raise RuntimeError(f"Failed to generated a unique name for '{name}'")


def build_unique_column_names(existing_names: List[AnyStr], column_prefix: AnyStr) -> NamedTuple:
    """Return a named tuple with prefixed API column names and their descriptions"""
    ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", API_COLUMN_NAMES_DESCRIPTION_DICT.keys())
    api_column_names = ApiColumnNameTuple(
        *[generate_unique(column_name, existing_names, column_prefix) for column_name in ApiColumnNameTuple._fields]
    )
    return api_column_names


def safe_json_loads(
    str_to_check: AnyStr, error_handling: ErrorHandling = ErrorHandling.LOG, verbose: bool = False,
) -> Dict:
    """Load a JSON string safely with an `error_handling` parameter"""
    if error_handling == ErrorHandling.FAIL:
        output = json.loads(str_to_check)
    else:
        try:
            output = json.loads(str_to_check)
        except (TypeError, ValueError):
            if verbose:
                logging.warning(f"Invalid JSON: '{str_to_check}'")
            output = {}
    return output


def move_api_columns_to_end(
    df: pd.DataFrame, api_column_names: NamedTuple, error_handling: ErrorHandling = ErrorHandling.LOG
) -> pd.DataFrame:
    """Move non-human-readable API columns to the end of the dataframe"""
    api_column_names_dict = api_column_names._asdict()
    if error_handling == ErrorHandling.FAIL:
        api_column_names_dict.pop("error_message", None)
        api_column_names_dict.pop("error_type", None)
    if not any(["error_raw" in column_name for column_name in df.keys()]):
        api_column_names_dict.pop("error_raw", None)
    columns = [column for column in df.keys() if column not in api_column_names_dict.values()]
    new_columns = columns + list(api_column_names_dict.values())
    df = df.reindex(columns=new_columns)
    return df

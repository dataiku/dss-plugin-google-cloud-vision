# -*- coding: utf-8 -*-
"""Module with read/write utility functions based on the Dataiku API"""

import os
from typing import Dict, AnyStr, List

import dataiku

import pandas as pd

from plugin_io_utils import PATH_COLUMN

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_path_df(folder: dataiku.Folder, file_extensions: List[AnyStr]) -> List[AnyStr]:
    """Generate a dataframe of file paths within a Dataiku Folder matching a list of extensions"""
    path_list = []
    if folder.read_partitions:
        for partition in folder.read_partitions:
            path_list += folder.list_paths_in_partition(partition)
    else:
        path_list = folder.list_paths_in_partition()
    filtered_path_list = [p for p in path_list if os.path.splitext(p)[1][1:].lower().strip() in file_extensions]
    if len(filtered_path_list) == 0:
        raise RuntimeError(f"No files detected with supported extensions '{file_extensions}', check input folder")
    path_df = pd.DataFrame(filtered_path_list, columns=[PATH_COLUMN])
    return path_df


def set_column_description(
    output_dataset: dataiku.Dataset, column_description_dict: Dict, input_dataset: dataiku.Dataset = None,
) -> None:
    """
    Set column descriptions of the output dataset based on a dictionary of column descriptions
    and retains the column descriptions from the input dataset (optional) if the column name matches.
    """
    if input_dataset is None:
        input_dataset_schema = []
    else:
        input_dataset_schema = input_dataset.read_schema()
    output_dataset_schema = output_dataset.read_schema()
    input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_description_dict.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)

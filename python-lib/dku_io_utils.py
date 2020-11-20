# -*- coding: utf-8 -*-
"""Module with read/write utility functions based on the Dataiku API"""

import os
from typing import Dict, AnyStr, List

import pandas as pd

import dataiku

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_path_df(folder: dataiku.Folder, file_extensions: List[AnyStr], path_column: AnyStr) -> pd.DataFrame:
    """Generate a dataframe of file paths in a Dataiku Folder matching a list of extensions

    Args:
        folder: Dataiku managed folder where files are stored
            This folder can be partitioned or not, this function handles both
        file_extensions: list of file extensions to match, ex: ["JPG", "PNG"]
            Expected format is not case-sensitive but should not include leading "."
        path_column: Name of the column in the output dataframe

    Returns:
        DataFrame with one column named `path_column` with all the file paths matching the list of `file_extensions`

    Raises:
        RuntimeError: If there are not files matching the list of `file_extensions`

    """
    path_list = []
    if folder.read_partitions:
        for partition in folder.read_partitions:
            path_list += folder.list_paths_in_partition(partition)
    else:
        path_list = folder.list_paths_in_partition()
    filtered_path_list = [
        path for path in path_list if os.path.splitext(path)[1][1:].lower().strip() in file_extensions
    ]
    if len(filtered_path_list) == 0:
        raise RuntimeError(f"No files detected with supported extensions {file_extensions}, check input folder")
    path_df = pd.DataFrame(filtered_path_list, columns=[path_column])
    return path_df


def set_column_description(
    output_dataset: dataiku.Dataset, column_description_dict: Dict, input_dataset: dataiku.Dataset = None,
) -> None:
    """Set column descriptions of the output dataset based on a dictionary of column descriptions

    Retains the column descriptions from the input dataset if the column name matches

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_description_dict: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions

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

# -*- coding: utf-8 -*-

"""
Input/Output plugin utility functions which *REQUIRE* the Dataiku API
"""

from typing import Dict, AnyStr, List, Callable
import pandas as pd

import dataiku

from plugin_io_utils import PATH_COLUMN


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_path_list(folder: dataiku.Folder) -> List[AnyStr]:
    partition = ""
    if folder.read_partitions is not None:
        partition = folder.read_partitions[0]
    path_list = folder.list_paths_in_partition(partition)
    assert len(path_list) != 0, "No files detected, check input folder"
    return path_list


def generate_path_df(folder: dataiku.Folder, path_filter_function: Callable) -> pd.DataFrame:
    path_list = [p for p in generate_path_list(folder) if path_filter_function(p)]
    assert len(path_list) != 0, "No files detected with supported extensions, check input folder"
    df = pd.DataFrame(path_list, columns=[PATH_COLUMN])
    return df


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

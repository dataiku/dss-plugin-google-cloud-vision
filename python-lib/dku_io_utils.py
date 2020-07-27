# -*- coding: utf-8 -*-
import dataiku
from typing import Dict, AnyStr, List, Callable

import pandas as pd


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

PATH_COLUMN = "path"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_image_uri(input_folder_bucket: AnyStr, input_folder_root_path: AnyStr, image_path: AnyStr) -> AnyStr:
    uri = "gs://{}/{}".format(input_folder_bucket, input_folder_root_path + image_path)
    return uri


def generate_path_list(folder: dataiku.Folder) -> List[AnyStr]:
    partition = ""
    if folder.read_partitions is not None:
        partition = folder.read_partitions[0]
    path_list = folder.list_paths_in_partition(partition)
    assert len(path_list) >= 1
    return path_list


def generate_path_df(folder: dataiku.Folder, path_filter_function: Callable) -> pd.DataFrame:
    image_path_list = [p for p in generate_path_list(folder) if path_filter_function(p)]
    df = pd.DataFrame(image_path_list, columns=[PATH_COLUMN])
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

# -*- coding: utf-8 -*-
"""Module with functions to parallelize API calls with error handling"""

import logging
import inspect
import math

from typing import Callable, AnyStr, List, Tuple, NamedTuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from time import time

import pandas as pd
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

from plugin_io_utils import ErrorHandling, build_unique_column_names


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_BATCH_SIZE = 10
DEFAULT_API_SUPPORT_BATCH = False
DEFAULT_VERBOSE = False


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class BatchAPIError(ValueError):
    """Custom exception raised if the Batch API fails"""

    pass


def api_call_single_row(
    api_call_function: Callable,
    api_column_names: NamedTuple,
    row: Dict,
    api_exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs,
) -> Dict:
    """Wrap a *single-row* API call function with error handling

    It takes the `api_call_function` function as input and:
    - ensures it has a `row` parameter which is a dict
    - parses the response to extract results and errors
    - handles errors from the function with two methods:
        * (default) log the error message as a warning and return the row with error keys
        * fail if there is an error
    """
    output_row = deepcopy(row)
    if error_handling == ErrorHandling.FAIL:
        response = api_call_function(row=row, **api_call_function_kwargs)
        output_row[api_column_names.response] = response
    else:
        for k in api_column_names:
            output_row[k] = ""
        try:
            response = api_call_function(row=row, **api_call_function_kwargs)
            output_row[api_column_names.response] = response
        except api_exceptions as e:
            logging.warning(f"API failed on: {row} because of error: {e}")
            error_type = str(type(e).__qualname__)
            module = inspect.getmodule(e)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            output_row[api_column_names.error_message] = str(e)
            output_row[api_column_names.error_type] = error_type
            output_row[api_column_names.error_raw] = str(e.args)
    return output_row


def api_call_batch(
    api_call_function: Callable,
    api_column_names: NamedTuple,
    batch: List[Dict],
    batch_api_response_parser: Callable,
    api_exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs,
) -> List[Dict]:
    """Wrap a *batch* API call function with error handling and response parsing

    It takes the `api_call_function` function as input and:
    - ensures it has a `batch` parameter which is a list of dict
    - parses the response to extract results and errors using the `batch_api_response_parser` function
    - handles errors from the function with two methods:
        * (default) log the error message as a warning and return the row with error keys
        * fail if there is an error
    """
    output_batch = deepcopy(batch)
    if error_handling == ErrorHandling.FAIL:
        response = api_call_function(batch=batch, **api_call_function_kwargs)
        output_batch = batch_api_response_parser(batch=batch, response=response, api_column_names=api_column_names)
        errors = [row[api_column_names.error_message] for row in batch if row[api_column_names.error_message] != ""]
        if len(errors) != 0:
            raise BatchAPIError(f"Batch API failed on: {batch} because of error: {errors}")
    else:
        try:
            response = api_call_function(batch=batch, **api_call_function_kwargs)
            output_batch = batch_api_response_parser(batch=batch, response=response, api_column_names=api_column_names)
        except api_exceptions as e:
            logging.warning(f"Batch API failed on: {batch} because of error: {e}")
            error_type = str(type(e).__qualname__)
            module = inspect.getmodule(e)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            for row in output_batch:
                row[api_column_names.response] = ""
                row[api_column_names.error_message] = str(e)
                row[api_column_names.error_type] = error_type
                row[api_column_names.error_raw] = str(e.args)
    return output_batch


def convert_api_results_to_df(
    input_df: pd.DataFrame,
    api_results: List[Dict],
    api_column_names: NamedTuple,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Combine API results (list of dict) with input dataframe

    Helper function to the `api_parallelizer` main function
    """
    if error_handling == ErrorHandling.FAIL:
        columns_to_exclude = [v for k, v in api_column_names._asdict().items() if "error" in k]
    else:
        columns_to_exclude = []
        if not verbose:
            columns_to_exclude = [api_column_names.error_raw]
    output_schema = {**{v: str for v in api_column_names}, **dict(input_df.dtypes)}
    output_schema = {k: v for k, v in output_schema.items() if k not in columns_to_exclude}
    record_list = [{col: result.get(col) for col in output_schema.keys()} for result in api_results]
    api_column_list = [c for c in api_column_names if c not in columns_to_exclude]
    output_column_list = list(input_df.columns) + api_column_list
    output_df = pd.DataFrame.from_records(record_list).astype(output_schema).reindex(columns=output_column_list)
    assert len(output_df.index) == len(input_df.index)
    return output_df


def api_parallelizer(
    input_df: pd.DataFrame,
    api_call_function: Callable,
    api_exceptions: Union[Exception, Tuple[Exception]],
    column_prefix: AnyStr,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    api_support_batch: bool = DEFAULT_API_SUPPORT_BATCH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs,
) -> pd.DataFrame:
    """Apply an API call function to a pandas.DataFrame with parallelization, batching, error handling and progress tracking

    The DataFrame is iterated on and passed to the function as dictionaries, row-by-row or by batches of rows.
    This iterative process is accelerated by the use of concurrent threads and is tracked with a progress bar.
    Errors are catched if they match the `api_exceptions` parameter and automatically logged.
    Once the whole DataFrame has been iterated on, API results and errors are added as additional columns.

    Args:
        input_df: Input dataframe which will be iterated on
        api_call_function: Function taking a dict as input and returning a dict
            If `api_support_batch` then the function works on list of dict
            Typically a function to call an API or do some enrichment
        api_exceptions: Tuple of Exception classes to catch
        column_prefix: Column prefix to add to the output columns for the API responses and errors
        parallel_workers: Number of concurrent threads
        api_support_batch: If True, send batches of row to the `api_call_function`
            Else (default) send rows as dict to the function
        batch_size: Number of rows to include in each batch
            Taken into account if `api_support_batch` is True
        error_handling: If ErrorHandling.LOG (default), log the error message as a warning
            and return the row with error keys.
            Else fail is there is any error.
        verbose: If True, log additional information on errors
            Else (default) log the error message and the error type
        **kwargs: Arbitrary keyword arguments passed to the `api_call_function`

    Returns:
        Input dataframe with additional columns:
        - API response from the `api_call_function`
        - API error message if any
        - API error type if any
    """
    df_iterator = (i[1].to_dict() for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    start = time()
    if api_support_batch:
        logging.info(f"Calling API endpoint with {len_iterator} rows, using batch size of {batch_size}...")
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    else:
        logging.info(f"Calling API endpoint with {len_iterator} rows...")
    api_column_names = build_unique_column_names(input_df.columns, column_prefix)
    pool_kwargs = api_call_function_kwargs.copy()
    more_kwargs = [
        "api_call_function",
        "error_handling",
        "api_exceptions",
        "api_column_names",
    ]
    for k in more_kwargs:
        pool_kwargs[k] = locals()[k]
    for k in ["fn", "row", "batch"]:  # Reserved pool keyword arguments
        pool_kwargs.pop(k, None)
    if not api_support_batch and "batch_api_response_parser" in pool_kwargs.keys():
        pool_kwargs.pop("batch_api_response_parser", None)
    api_results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        if api_support_batch:
            futures = [pool.submit(api_call_batch, batch=batch, **pool_kwargs) for batch in df_iterator]
        else:
            futures = [pool.submit(api_call_single_row, row=row, **pool_kwargs) for row in df_iterator]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            api_results.append(f.result())
    if api_support_batch:
        api_results = flatten(api_results)
    output_df = convert_api_results_to_df(input_df, api_results, api_column_names, error_handling, verbose)
    num_api_error = sum(output_df[api_column_names.response] == "")
    num_api_success = len(input_df.index) - num_api_error
    logging.info(
        (
            f"Calling API endpoint: {num_api_success} rows succeeded, {num_api_error} failed "
            f"in {(time() - start):.2f} seconds."
        )
    )
    return output_df

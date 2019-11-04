import json
import os
import logging
from google.oauth2 import service_account

ALLOWED_FORMATS = ['jpeg', 'jpg', 'png']

def get_credentials(connection_info):
    print("######"+ json.dumps(connection_info))
    if connection_info.get("credentials") is None:
        return None
    try:
        obj = json.loads(connection_info.get("credentials"))
    except Exception as e:
        logging.error(e)
        raise ValueError("Provided credentials are not JSON")
    credentials = service_account.Credentials.from_service_account_info(obj)
    _log_get_credentials(credentials)
    return credentials

def _log_get_credentials(credentials):
    if hasattr(credentials, 'service_account_email'):
        logging.info("Credentials loaded : %s" % credentials.service_account_email)
    else:
        logging.info("Credentials loaded")

def get_full_file_path(file_name=None, folder=None):
    """
    Return the absolute path to a file located in a managed folder.
    Optionally perform file type check.
    """
    ext = file_name.split(".")[-1]
    if ext not in ALLOWED_FORMATS:
        raise ValueError("Invalid file format ({})".format(ext))
    else:
        folder_path = folder.get_path()
        full_path = folder_path + "/" + file_name
        return full_path


def suffix_filename(file=None, suffix=None):
    file_split = file.split(".")
    res = file_split[0] + "_" + suffix + "." + file_split[-1]
    return res

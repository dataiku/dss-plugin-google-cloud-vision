import dataiku
import logging
import io
import os
import json
from dataiku.customrecipe import *
from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.json_format import MessageToJson
from misc_helpers import get_credentials

#==============================================================================
# SETUP
#==============================================================================

ALLOWED_FORMATS = ['jpeg', 'jpg', 'png']
logging.basicConfig(level=logging.INFO, format='[Google Cloud Vision Plugin] %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

recipe_config = get_recipe_config()
connectionInfo = recipe_config.get("connection_info")

input_folder_name = get_input_names_for_role("input_folder")[0]
input_folder = dataiku.Folder(input_folder_name)
output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

credentials = get_credentials(connectionInfo)
client = vision.ImageAnnotatorClient(credentials=credentials)

#==============================================================================
# RUN
#==============================================================================

output_schema = [
    {"name": "source", "type": "string"},
    {"name": "predicted_labels", "type": "string"},
    # {"name": "prediction_confidence", "type": "float"},
    # {"name": "topicality", "type": "float"},
    # {"name": "mid", "type": "string"}
]
output_dataset.write_schema(output_schema)


with output_dataset.get_writer() as writer:
    max_results = int(recipe_config.get("output_max_nb"))
    input_folder_path = input_folder.get_path()
    for img_file in os.listdir(input_folder_path):
        if img_file.split(".")[-1] not in ALLOWED_FORMATS:
            raise ValueError("Invalid image format")
        img_full_path = input_folder_path + '/' + img_file
        logging.info("Processing image {}".format(img_full_path))
        with io.open(img_full_path, 'rb') as f:
            img_content = f.read()
        img_stream = types.Image(content=img_content)
        response = client.label_detection(image=img_stream, max_results=max_results)
        resp_json = json.loads(MessageToJson(response))
        annotations = resp_json.get("labelAnnotations")
        output_row = {}
        output_row["source"] = img_file
        output_row["predicted_labels"] = resp_json.get("labelAnnotations")
        # output_row["prediction_confidence"] = ann["score"]
        # output_row["topicality"] = ann["topicality"]
        # output_row["mid"] = ann["mid"]
        writer.write_row_dict(output_row)

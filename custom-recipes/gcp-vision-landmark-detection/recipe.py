import dataiku
import logging
import json
from PIL import Image
from dataiku.customrecipe import *
from dku_gcp_vision import *
from bbox import draw_bounding_boxes

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[Google Cloud Vision Plugin] %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

connection_info = get_recipe_config().get("connection_info")
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_folder_name = get_input_names_for_role("input_folder")[0]
input_folder = dataiku.Folder(input_folder_name)

should_output_json = len(get_input_names_for_role('input_folder')) > 0
if should_output_json:
    output_dataset_name = get_output_names_for_role('output_dataset')[0]
    output_dataset = dataiku.Dataset(output_dataset_name)

should_draw_bbox = len(get_output_names_for_role('output_folder')) > 0
if should_draw_bbox:
    output_folder_name = get_output_names_for_role('output_folder')[0]
    output_folder = dataiku.Folder(output_folder_name)
    output_folder_path = output_folder.get_path()

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

if should_output_json:
    output_schema = [
        {"name": "file_path", "type": "string"},
        {"name": "detected_landmarks", "type": "string"}
    ]
    if should_output_raw_results:
        output_schema.append({"name": "raw_results", "type": "string"})
    output_dataset.write_schema(output_schema)
    writer = output_dataset.get_writer()

for filepath in os.listdir(input_folder.get_path()):
    if supported_image_format(filepath):
        with open(os.path.join(input_folder.get_path(), filepath), "rb") as image_file:
            row, response, bbox_list = detect_landmarks(image_file, client)
            if should_output_raw_results:
                row["raw_results"] = json.dumps(response, default=lambda x: x.__dict__)
            if should_draw_bbox:
                image_with_bounding_boxes = draw_bounding_boxes(Image.open(image_file), bbox_list)
                image_with_bounding_boxes.save(os.path.join(output_folder_path, filepath))
    else:
        logging.warn("Cannot score file (only JPEG, JPG and PNG extension are supported): " + filepath)
        row = {}
    row["file_path"] = filepath
    if should_output_json:
        writer.write_row_dict(row)

if should_output_json:
    writer.close()

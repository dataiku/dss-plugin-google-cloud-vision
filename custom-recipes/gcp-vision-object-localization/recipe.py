import dataiku
import logging
import io
import json
from dataiku.customrecipe import *
from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.json_format import MessageToJson
from misc_helpers import get_credentials
from misc_helpers import suffix_filename
from vision_helpers import draw_bounding_boxes, dedup_bounding_boxes

#==============================================================================
# SETUP
#==============================================================================

ALLOWED_FORMATS = ['jpeg', 'jpg', 'png']

logging.basicConfig(level=logging.INFO, format='[Google Cloud Vision Plugin] %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

connectionInfo = get_recipe_config().get("connection_info")
img_new_suffix = get_recipe_config().get("labelled_img_suffix")
with_bboxes = get_recipe_config().get("with_bounding_boxes")

input_folder_name = get_input_names_for_role("input_folder")[0]
input_folder = dataiku.Folder(input_folder_name)
output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if with_bboxes and len(get_output_names()) > 1:
    output_folder_name = get_output_names_for_role("output_folder")[0]
    output_folder = dataiku.Folder(output_folder_name)

credentials = get_credentials(connectionInfo)
client = vision.ImageAnnotatorClient(credentials=credentials)

#==============================================================================
# RUN
#==============================================================================

output_schema = [{"name": "source", "type": "string"}, {"name": "objects", "type": "string"}]
output_dataset.write_schema(output_schema)

writer = output_dataset.get_writer()
input_folder_path = input_folder.get_path()
for img_file in os.listdir(input_folder_path):
    if img_file.split(".")[-1] not in ALLOWED_FORMATS:
        raise ValueError("Invalid image format")
    img_full_path = input_folder_path + "/" + img_file
    with io.open(img_full_path, 'rb') as f:
        img_content = f.read()
    img_stream = types.Image(content=img_content)
    response = client.object_localization(image=img_stream)
    response_json = json.loads(MessageToJson(response))
    objects = [{"label": x["name"],
                "score": '%.2f'%(x["score"]),
                "vertices": x["boundingPoly"]["normalizedVertices"]} \
                    for x in response_json["localizedObjectAnnotations"]]
    objects_dedup = dedup_bounding_boxes(object_list=objects)

    output_row = {}
    output_row["source"] = img_file
    output_row["objects"] = objects_dedup
    writer.write_row_dict(output_row)
    if with_bboxes:
        output_img = {}
        output_img["source"] = img_file
        output_img["full_path"] = img_full_path
        output_img["objects"] = objects_dedup
        draw_bounding_boxes(img_dict=output_img,
                            output_folder=output_folder,
                            output_file_suffix=img_new_suffix,
                            deduplicate=False)
writer.close()

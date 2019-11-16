import logging
import hashlib
import json
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

SUPPORTED_IMAGE_FORMATS = ['jpeg', 'jpg', 'png']

def get_client(connection_info):
    credentials = _get_credentials(connection_info)
    return vision.ImageAnnotatorClient(credentials=credentials)

def supported_image_format(filepath):
    extension = filepath.split(".")[-1].lower()
    return extension in SUPPORTED_IMAGE_FORMATS

def detect_adult_content(image_file, client, max_results=None):
    img_stream = types.Image(content=image_file.read())
    response = client.safe_search_detection(image=img_stream)
    resp_json = json.loads(MessageToJson(response))
    row = {}
    v = resp_json.get("safeSearchAnnotation")
    if v:
        row["is_adult_content"] = v["adult"].lower().replace("_", " ")
        row["is_suggestive_content"] = v["racy"].lower().replace("_", " ")
        row["is_violent_content"] = v["violence"].lower().replace("_", " ")
    return row, resp_json

def detect_labels(image_file, client, max_results=None):
    img_stream = types.Image(content=image_file.read())
    response = client.label_detection(image=img_stream, max_results=max_results)
    resp_json = json.loads(MessageToJson(response))
    annotations = resp_json.get("labelAnnotations")
    row = {}
    labels = [x["description"] for x in resp_json.get("labelAnnotations")]
    if len(labels):
        row["predicted_labels"] = json.dumps(labels)
    return row, resp_json

def detect_brands(image_file, client):
    row = {}
    img_stream = types.Image(content=image_file.read())
    response = client.logo_detection(image=img_stream)
    response_json = json.loads(MessageToJson(response))
    def make_bbox(x):
        v = x["boundingPoly"]["vertices"]
        try:
            return {
                "label": x["description"],
                "score": '%.2f'%(x["score"]),
                "top": v[0]['y'],
                "left": v[0]['x'],
                "width": v[1]['x'] - v[0]['x'],
                "height": v[3]['y'] - v[0]['y']
            }
        except Exception as e:
            # For some reason, x or y can be undefined
            logging.info(e)
            return None
    bbox_list = []
    if "logoAnnotations" in response_json:
        bbox_list = [ make_bbox(x) for x in response_json["logoAnnotations"]]
        bbox_list = [x for x in bbox_list if x]
        # bbox_list = dedup_bounding_boxes(objects_list=bbox_list)
        if len(bbox_list):
            row["detected_logos"] = json.dumps(bbox_list)
    return row, response_json, bbox_list


def detect_objects(image_file, client):
    row = {}
    img_stream = types.Image(content=image_file.read())
    response = client.object_localization(image=img_stream)
    response_json = json.loads(MessageToJson(response))
    def make_bbox(x):
        v = x["boundingPoly"]["normalizedVertices"]
        try:
            return {
                "label": x["name"],
                "score": '%.2f'%(x["score"]),
                "top": v[0]['y'],
                "left": v[0]['x'],
                "width": v[1]['x'] - v[0]['x'],
                "height": v[3]['y'] - v[0]['y']
            }
        except Exception as e:
            # For some reason, x or y can be undefined
            logging.info(e)
            return None
    bbox_list = []
    if "localizedObjectAnnotations" in response_json:
        bbox_list = [ make_bbox(x) for x in response_json["localizedObjectAnnotations"]]
        bbox_list = [x for x in bbox_list if x]
        # bbox_list = dedup_bounding_boxes(objects_list=bbox_list)
        if len(bbox_list):
            row["detected_objects"] = json.dumps(bbox_list)
    return row, response_json, bbox_list

def detect_landmarks(image_file, client):
    row = {}
    img_stream = types.Image(content=image_file.read())
    response = client.landmark_detection(image=img_stream)
    response_json = json.loads(MessageToJson(response))
    def make_bbox(x):
        v = x["boundingPoly"]["vertices"]
        try:
            return {
                "label": x["description"],
                "score": '%.2f'%(x["score"]),
                "top": v[0]['y'],
                "left": v[0]['x'],
                "width": v[1]['x'] - v[0]['x'],
                "height": v[3]['y'] - v[0]['y']
            }
        except Exception as e:
            # For some reason, x or y can be undefined
            logging.info(e)
            return None
    bbox_list = []
    if "landmarkAnnotations" in response_json:
        bbox_list = [ make_bbox(x) for x in response_json["landmarkAnnotations"]]
        bbox_list = [x for x in bbox_list if x]
        # bbox_list = dedup_bounding_boxes(objects_list=bbox_list)
        if len(bbox_list):
            row["detected_landmarks"] = json.dumps(bbox_list)
    return row, response_json, bbox_list

# def denormalize_vertices(vertices=None, width=None, height=None):
#     """
#     Translate vertices coordinates back from [0,1]x[0,1] to the original pixel
#     space.
#     """
#     vertices_d = []
#     for pt in vertices:
#         for axis in ["x", "y"]:
#             if axis not in pt.keys():
#                 pt[axis] = 0.0
#         pt_d = (width*pt["x"], height*pt["y"])
#         vertices_d.append(pt_d)
#     return vertices_d


def dedup_bounding_boxes(objects_list=None):
    """
    Remove duplicate bounding boxes designating same object with distinct labels
    (e.g. 'fruit' and 'apple' or 'furniture' and 'chair')
    """
    nb_objects = len(objects_list)
    if nb_objects == 1:
        logging.info("DEDUP - Singleton, no dedup required")
        return None
    obj_hashes = []
    d = {}
    for obj in objects_list:
        # Hash the coordinates of bounding boxes and use them as a key for deduplication
        obj_hash = hashlib.md5(str(obj["vertices"]).encode(encoding="utf-8")).hexdigest()
        if obj_hash not in d:
            d[obj_hash] = []
        d[obj_hash].append(obj)
    # Deduplicate by only keeping the highest score
    d_sorted = {k: sorted(v, key=lambda x: x['score'], reverse=True) for (k,v) in d.items()}
    d_dedup = list({k: v[0] for (k,v) in d_sorted.items()}.values())
    return d_dedup


def _get_credentials(connection_info):
    if connection_info.get("private_key") is None or len(connection_info.get("private_key")) == 0:
        return None
    try:
        private_key = json.loads(connection_info.get("private_key"))
    except Exception as e:
       logging.error(e)
       raise ValueError("Provided credentials are not JSON")
    credentials = service_account.Credentials.from_service_account_info(private_key)
    if hasattr(credentials, 'service_account_email'):
        logging.info("Credentials loaded : %s" % credentials.service_account_email)
    else:
        logging.info("Credentials loaded")
    return credentials

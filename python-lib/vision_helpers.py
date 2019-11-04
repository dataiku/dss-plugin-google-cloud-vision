import math
import logging
import hashlib

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
from matplotlib.colors import rgb2hex
from ast import literal_eval
from misc_helpers import get_full_file_path, suffix_filename

DEFAULT_FONT = "Arial.ttf"

logging.basicConfig(level=logging.INFO, format='GCAI plugin %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def denormalize_vertices(vertices=None, width=None, height=None):
    """
    Translate vertices coordinates back from [0,1]x[0,1] to the original pixel
    space.
    """
    vertices_d = []
    for pt in vertices:
        for axis in ["x", "y"]:
            if axis not in pt.keys():
                pt[axis] = 0.0
        pt_d = (width*pt["x"], height*pt["y"])
        vertices_d.append(pt_d)
    return vertices_d


def dedup_bounding_boxes(object_list=None):
    """
    Remove duplicate bounding boxes designating same object with distinct labels
    (e.g. 'fruit' and 'apple' or 'furniture' and 'chair')
    """
    nb_objects = len(object_list)
    if nb_objects == 1:
        logging.info("DEDUP - Singleton, no dedup required")
        return None
    obj_hashes = []
    d = {}
    for obj in object_list:
        # Hash the coordinates of bounding boxes and use them as a key for deduplication
        obj_hash = hashlib.md5(str(obj["vertices"]).encode(encoding="utf-8")).hexdigest()
        if obj_hash not in d:
            d[obj_hash] = []
        d[obj_hash].append(obj)
    # Deduplicate by only keeping the highest score
    d_sorted = {k: sorted(v, key=lambda x: x['score'], reverse=True) for (k,v) in d.items()}
    d_dedup = list({k: v[0] for (k,v) in d_sorted.items()}.values())
    return d_dedup


def draw_bounding_boxes(img_dict=None, output_folder=None, output_file_suffix=None, deduplicate=False):
    """
    Draw rectangular boxes around detected objects in an image.
    """
    img_path = img_dict["full_path"]
    logging.info("Opening image {}...".format(img_path))
    try:
        img = Image.open(img_path)
        width = img.width
        height = img.height
        draw = ImageDraw.Draw(img)
    except:
        raise ValueError("Invalid input image")
        return
    if img_dict["objects"] is None:
        return img
    logging.info("Successfully opened {} ({},{})".format(img_path, width, height))
    # Scale bounding box elements:
    bbox_line_size = math.floor(width/100)
    score_text_size = math.floor(width/50)
    font = ImageFont.truetype(DEFAULT_FONT, size=int(score_text_size))
    # Loop
    # -- If drawn from a dataset with labels, convert string to list:
    if not isinstance(img_dict["objects"], list):
        logging.info("Converting object data from STRING to LIST...")
        import json
        print("###@#"+ json.dumps(img_dict))
        objects = literal_eval(img_dict["objects"])
    else:
        objects = img_dict["objects"]
    bbox_list = []
    for (cpt, obj) in enumerate(objects):
        obj_formatted_label = obj["label"] + "({})".format(obj["score"])
        bbox_coord = denormalize_vertices(vertices=obj["vertices"], width=width, height=height)
        bbox_list.append({"fmt_label": obj_formatted_label, "coord": bbox_coord, "color": rgb2hex(cm.tab10(cpt))})

    # Draw
    for bbox in bbox_list:
        logging.info("Drawing {}".format(bbox["fmt_label"]))
        draw.rectangle((bbox["coord"][3], bbox["coord"][1]), width=int(bbox_line_size), outline=bbox["color"])
        offset = score_text_size + bbox_line_size
        draw.text((bbox["coord"][3][0]+offset, bbox["coord"][3][1]-1.5*offset),
                  text=bbox["fmt_label"],
                  fill=bbox["color"],
                  font=font)
    # Save image to output folder
    img_output = suffix_filename(file=img_dict["source"], suffix=output_file_suffix)
    img_output_full_path = get_full_file_path(file_name=img_output, folder=output_folder)
    img.save(img_output_full_path)
    img.close()

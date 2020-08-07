# -*- coding: utf-8 -*-

"""
Utility functions to manipulate images
"""

import os
from typing import List, AnyStr, Dict

import numpy as np
from dataiku.customrecipe import get_recipe_resource
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

BOUNDING_BOX_COLOR = "red"
try:
    BOUNDING_BOX_FONT_PATH = os.path.join(get_recipe_resource(), "SourceSansPro-Regular.ttf")
except TypeError:
    BOUNDING_BOX_FONT_PATH = ImageFont.load_default()
BOUNDING_BOX_FONT_DEFAULT_SIZE = 18


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def save_image_bytes(pil_image: Image, path: AnyStr) -> bytes:
    image_bytes = BytesIO()
    file_extension = path.split(".")[-1].lower()
    if file_extension in {"jpg", "jpeg"}:
        pil_image.save(
            image_bytes,
            format="JPEG",
            quality=100,
            exif=pil_image.getexif(),
            icc_profile=pil_image.info.get("icc_profile"),
        )
    elif file_extension == "png":
        pil_image.save(image_bytes, format="PNG", optimize=True)
    elif file_extension == "tiff" or file_extension == "tif":
        pil_image.save(image_bytes, format="TIFF", save_all=True)
    else:
        pil_image.save(image_bytes, format=file_extension)
    return image_bytes


def scale_bounding_box_font(image: Image, text_line_list: List[AnyStr], bbox_left: int, bbox_right: int) -> ImageFont:
    """
    Compute font for bounding box text to enforce specific design guidelines for short text:
    - use custom font from BOUNDING_BOX_FONT_PATH (bundled in the resource folder of the plugin)
    - scale font size to fit the text width to percentages of the width of the image and bounding box
      and avoid text overflowing to the right outside the image
    - bucket font size in increments (4, 6, 8, ...) to homogenize font sizing
    Note that this function is designed for languages which read horizontally from left to right
    """
    # Initialize font
    im_width, im_height = image.size
    font_default_size = ImageFont.truetype(font=BOUNDING_BOX_FONT_PATH, size=BOUNDING_BOX_FONT_DEFAULT_SIZE)
    text_width_default_size = max([font_default_size.getsize(t)[0] for t in text_line_list])
    # Scale font size to percentages of the width of the image and bounding box
    target_width = int(max(0.2 * im_width, 0.4 * (bbox_right - bbox_left)))
    if bbox_left + target_width > im_width:
        target_width = int(im_width - bbox_left)
    scaled_font_size = int(target_width * BOUNDING_BOX_FONT_DEFAULT_SIZE / text_width_default_size)
    scaled_font = font_default_size.font_variant(size=scaled_font_size)
    # Bucket font size in increments (2, 4, 6, 8, ...) to homogenize font sizing
    scaled_font_size = max(2 * int(np.ceil(scaled_font_size / 2.0)), 4)
    scaled_font = font_default_size.font_variant(size=scaled_font_size)
    return scaled_font


def draw_bounding_box_pil_image(
    image: Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    text: AnyStr = "",
    use_normalized_coordinates: bool = True,
    color: AnyStr = BOUNDING_BOX_COLOR,
):
    """
    Draws a bounding box on an image, assuming parallel lines to the image orientation.
    Inspired by https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the 'use_normalized_coordinates' argument.
    Text is displayed on a separate line above the bounding box in black text on a rectangle filled with 'color'.
    If the top of the bounding box extends to the edge of the image, text is displayed below the bounding box.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        text: strings to display in box.
        color: color to draw bounding box and text rectangle. Default is BOUNDING_BOX_COLOR.
        use_normalized_coordinates: If True (default), treat coordinates as relative to the image.
            Otherwise treat coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    line_thickness = 3 * int(np.ceil(0.001 * max(im_width, im_height)))
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    lines = [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
    draw.line(xy=lines, width=line_thickness, fill=color)
    if text != "" and text is not None:
        text_line_list = text.splitlines()
        scaled_font = scale_bounding_box_font(image, text_line_list, left, right)
        # If the total height of the display strings added to the top of the bounding box
        # exceeds the top of the image, stack the strings below the bounding box instead of above.
        text_height = sum([scaled_font.getsize(t)[1] for t in text_line_list])
        text_height_with_margin = (1 + 2 * 0.05) * text_height  # Each line has a top and bottom margin of 0.05x
        text_bottom = top
        if top < text_height_with_margin:
            text_bottom += text_height_with_margin
        # Reverse list and print from bottom to top.
        for t in text_line_list[::-1]:
            text_width, text_height = scaled_font.getsize(t)
            margin = int(np.ceil(0.05 * text_height))
            rectangle = [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom + 2 * margin)]
            draw.rectangle(xy=rectangle, fill=color)
            draw.text(xy=(left + margin, text_bottom - text_height - margin), text=t, fill="black", font=scaled_font)
            text_bottom -= text_height - 2 * margin


def draw_bounding_poly_pil_image(
    image: Image, vertices: List[Dict], color: AnyStr = BOUNDING_BOX_COLOR,
):
    """
    Draws a bounding polygon on an image, with lines which may not be parallel to the image orientaiton.
    Vertices must be specified in absolute (pixel) coordinates.

    Args:
        image: a PIL.Image object.
        vertices: List of {"x": 2, "y": 4} dictionaries for absolute coordinates
        color: color to draw bounding box and text rectangle. Default is BOUNDING_BOX_COLOR.
    """
    draw = ImageDraw.Draw(image)
    if len(vertices) == 4:
        draw.polygon(
            xy=[
                (vertices[0].get("x", 0), vertices[0].get("y", 0)),
                (vertices[1].get("x", 0), vertices[1].get("y", 0)),
                (vertices[2].get("x", 0), vertices[2].get("y", 0)),
                (vertices[3].get("x", 0), vertices[3].get("y", 0)),
            ],
            fill=None,
            outline=color,
        )
    else:
        raise ValueError("Bounding polygon does not contain 4 vertices: {}".format(vertices))


def crop_pil_image(
    image: Image, ymin: float, xmin: float, ymax: float, xmax: float, use_normalized_coordinates: bool = True,
):
    """
    Crops an image given an bounding box
    """
    im_width, im_height = image.size
    box = (xmin, ymin, xmax, ymax)
    if use_normalized_coordinates:
        box = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
    image = image.crop(box)
    return image

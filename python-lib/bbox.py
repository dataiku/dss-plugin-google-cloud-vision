from PIL import ImageDraw, ImageFont

DEFAULT_FONT = "Arial.ttf"

"""
bbox_list: list of {label, score, top, left, width, height}
  * score is supposed to be in 0-1 range
  * top, left, width, height are normalized (0-1 range)
"""
def draw_bounding_boxes(pil_image, bbox_list):
    output_image = pil_image.convert(mode='RGB')
    width, height = pil_image.size
    draw = ImageDraw.Draw(output_image)

    for bbox in bbox_list:
        left = width * bbox['left']
        top = height * bbox['top']
        w = width * bbox['width']
        h = height * bbox['height']
        draw.rectangle([left,top, left + w, top + h], outline='#00FF00')
        draw.text([left, top], bbox['label']) # add proba and improve font

    return output_image

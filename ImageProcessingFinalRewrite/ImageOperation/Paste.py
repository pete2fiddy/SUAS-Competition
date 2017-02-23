
from PIL import Image
def paste_img_onto_img(paste_img, base_img, offset = None):
    base_img.paste(paste_img, box=offset)
    return None
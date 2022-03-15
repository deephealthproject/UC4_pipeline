import matplotlib.pyplot as plt
import pydicom
import png
import cv2
import numpy as np
from PIL import Image
from pydicom.data import get_testdata_files
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

path_im1 = str(BASE_DIR.joinpath('data/raw/IMG168.png'))
path_im2 = str(BASE_DIR.joinpath('data/raw/aliza_001.png'))
output_path = str(BASE_DIR.joinpath('data/raw/diff.png'))
im1 = cv2.imread(path_im1, -cv2.IMREAD_ANYDEPTH)
im2 = cv2.imread(path_im2, -cv2.IMREAD_ANYDEPTH)
print(im1.shape)
print(im2.shape)
print(im1.dtype)
print(im2.dtype)
diff = im1 - im2
print(diff[diff != 0].shape)
print((im1==im2).all())

background = Image.open(path_im1)
overlay = Image.open(path_im2)

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.2)
new_img.save(output_path,"PNG")

def load_uint16_png(path):
    reader = png.Reader(path)
    pngdata = reader.read()
    px_array = np.array(map(np.uint16, pngdata[2]))
    print(px_array.dtype)
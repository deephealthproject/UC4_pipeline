# import matplotlib.pyplot as plt
# import pydicom
# import png
# import cv2
# import numpy as np
# from PIL import Image
# from pydicom.data import get_testdata_files
# from pydicom.pixel_data_handlers.util import apply_modality_lut
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# from pathlib import Path

# def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
#     '''
#     Map a 16-bit image trough a lookup table to convert it to 8-bit.

#     Parameters
#     ----------
#     img: numpy.ndarray[np.uint16]
#         image that should be mapped
#     lower_bound: int, optional
#         lower bound of the range that should be mapped to ``[0, 255]``,
#         value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
#         (defaults to ``numpy.min(img)``)
#     upper_bound: int, optional
#        upper bound of the range that should be mapped to ``[0, 255]``,
#        value must be in the range ``[0, 65535]`` and larger than `lower_bound`
#        (defaults to ``numpy.max(img)``)

#     Returns
#     -------
#     numpy.ndarray[uint8]
#     '''
#     if not(0 <= lower_bound < 2**16) and lower_bound is not None:
#         raise ValueError(
#             '"lower_bound" must be in the range [0, 65535]')
#     if not(0 <= upper_bound < 2**16) and upper_bound is not None:
#         raise ValueError(
#             '"upper_bound" must be in the range [0, 65535]')
#     if lower_bound is None:
#         lower_bound = np.min(img)
#     if upper_bound is None:
#         upper_bound = np.max(img)
#     if lower_bound >= upper_bound:
#         raise ValueError(
#             '"lower_bound" must be smaller than "upper_bound"')
#     lut = np.concatenate([
#         np.zeros(lower_bound, dtype=np.uint16),
#         np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
#         np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
#     ])
#     return lut[img].astype(np.uint8)

# BASE_DIR = Path(__file__).resolve().parents[2]

# path = BASE_DIR.joinpath('data/raw/IMG168')
# mask_path = BASE_DIR.joinpath('data/raw/deephealth_molinette_001_e1_168_001.png')

# ds = pydicom.dcmread(str(path))
# arr = ds.pixel_array
# shape = arr.shape
# print(shape)
# print(arr.dtype) #int16
# print(arr.min())
# print(arr.max())
# hu = apply_modality_lut(arr, ds) #float64
# print(hu.min())
# print(hu.max())
# out = apply_voi_lut(hu, ds, index=0)
# print(out.dtype) #float64
# # Convert to uint
# image_2d_scaled = out

# #im = Image.fromarray(image_2d_scaled)
# output_path = BASE_DIR.joinpath('data/raw/IMG168.png')
# #im.save(output_path)


# print(image_2d_scaled.min(), image_2d_scaled.max())
# z16 = (65535*((image_2d_scaled - image_2d_scaled.min())/image_2d_scaled.ptp())).astype(np.uint16) #change range values to [0, 65535] and then change type to uint16
# print(z16)
# print(z16.min(), z16.max())
# z16 = (np.maximum(z16,0) / z16.max()) * 255.0 # feature scaling, from [0, 65535] to [0, 255]
# print(np.maximum([0,2,3,4,0],0))
# z8 = z16.astype(np.uint8) # change type

# print(z16.min(), z16.max())
# print(z8.min(), z8.max())
# # Write the PNG file
# with open('data/raw/IMG168.png', 'wb') as png_file:
#     w = png.Writer(shape[1], shape[0], greyscale=True, bitdepth=8)
#     w.write(png_file, z8)

# mask_image = cv2.imread(str(mask_path))
# #plt.imsave(output_path, out, cmap='gray')

# fig = plt.figure(frameon=False)
# fig.set_size_inches(1, 1)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)

# plt.imshow(z8, cmap='gray', aspect='auto')
# plt.imshow(mask_image, alpha=0.8, aspect='auto')
# plt.savefig('data/raw/mask.png', dpi=512)
# plt.show()

import pydicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

MIN_BOUND = -1200.0
MAX_BOUND = 600.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def rescaling(x, new_min, new_max):
    return new_min + ((x - x.min())*(new_max - new_min))/(x.max()-x.min())

# python -m visualization.visualize_dicom
if __name__ == '__main__':
    input_path = 'data/raw/IMG168'
    #output_path = 'data/raw/IMG1.png'
    mask_path = 'data/raw/deephealth_molinette_001_e1_168_001.png'
    mask_image = cv2.imread(str(mask_path))

    # Read raw dicom file
    dcm = pydicom.dcmread(input_path)
    #print(image_dcm)

    # Get pixels
    img = dcm.pixel_array #int16

    # Intensity values of -2000 are the pixels that fall outside of the scanner bounds.
    img[img == -2000] = 0

    # Convert to Hounsfield units (HU)
    img = apply_modality_lut(img, dcm) #float64

    img = apply_voi_lut(img, dcm, index=0)

    #img = normalize(img)

    # Rescale image and convert
    img = rescaling(img, 0, 65535)
    img = np.uint16(img)
    img = rescaling(img, 0, 255.0) 
    img = np.uint8(img)

    fig = plt.figure(frameon=False)
    #fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(img, cmap='gray')
    plt.imshow(mask_image, alpha=0.8)
    #plt.savefig('data/raw/mask.png', dpi=512)
    plt.show()

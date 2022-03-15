import numpy as np
import png
import pydicom
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.pixel_data_handlers.util import apply_voi_lut
from .preprocessing import rescaling, get_segmented_lungs

 # Some infos on dicom format https://www.codeproject.com/Articles/36014/DICOM-Image-Viewer
def dicom_to_png(input_path, output_path=None, hu=True, windowing=True, segment=False):

    # Read raw dicom file
    dcm = pydicom.dcmread(str(input_path))
    dcm.WindowCenter = -500
    dcm.WindowWidth = 1600
    #print(image_dcm)

    # Get pixels
    img = dcm.pixel_array #int16

    # Intensity values of -2000 are the pixels that fall outside of the scanner bounds.
    img[img == -2000] = 0

    # Convert to Hounsfield units (HU)
    if hu:
        img = apply_modality_lut(img, dcm) #float64
    
    # Get segmented lungs
    if segment:
        img = get_segmented_lungs(img)
    
    img_default = np.copy(img)
    #img_lungs = np.copy(img)
    #img_mediastinal = np.copy(img)

    # Apply windowing
    if windowing:
        img_default = apply_voi_lut(img_default, dcm, index=0) #float64
        # dcm.WindowCenter = 50
        # dcm.WindowWidth = 350
        # img_lungs = apply_voi_lut(img_lungs, dcm, index=0) #float64
        # dcm.WindowCenter = -500
        # dcm.WindowWidth = 1600
        # img_mediastinal = apply_voi_lut(img_mediastinal, dcm, index=0) #float64

    # Rescale image and convert
    img_default = rescaling(img_default, 0, 65535)
    img_default = np.uint16(img_default)
    img_default = rescaling(img_default, 0, 255.0) 
    img_default = np.uint8(img_default)

    # img_lungs = rescaling(img_lungs, 0, 65535)
    # img_lungs = np.uint16(img_lungs)
    # img_lungs = rescaling(img_lungs, 0, 255.0) 
    # img_lungs = np.uint8(img_lungs)

    # img_mediastinal = rescaling(img_mediastinal, 0, 65535)
    # img_mediastinal = np.uint16(img_mediastinal)
    # img_mediastinal = rescaling(img_mediastinal, 0, 255.0) 
    # img_mediastinal = np.uint8(img_mediastinal)

    #output_image = np.stack((img_lungs, img_mediastinal, img_default))
    #print(output_image.shape)

    # Write the PNG file
    img = img_default
    if output_path:
      with open(output_path, 'wb') as png_file:
           w = png.Writer(img.shape[1], img.shape[0], greyscale=True, bitdepth=8)
           w.write(png_file, img)

    # with open(output_path, 'wb') as out_file:
    #     np.save(out_file, output_image)

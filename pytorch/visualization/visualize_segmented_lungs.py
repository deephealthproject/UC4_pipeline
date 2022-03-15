import matplotlib.pyplot as plt
import pydicom
from utils.dicom_utils import dicom_to_png
from utils.preprocessing import get_segmented_lungs
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# python -m visualization.visualize_segmented_lungs
if __name__ == '__main__':
    input_path = 'data/raw/IMG168'
    #output_path = 'data/raw/IMG1.png'
    mask_path = 'data/raw/deephealth_molinette_001_e1_168_001.png'

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
    
    # Get segmented lungs
    seg_img = get_segmented_lungs(img, plot=False)

    _, plots = plt.subplots(1, 2)
    plots[0].axis('off')
    plots[0].imshow(img, cmap=plt.cm.bone)
    plots[1].axis('off')
    plots[1].imshow(seg_img, cmap=plt.cm.bone)
    plt.show()

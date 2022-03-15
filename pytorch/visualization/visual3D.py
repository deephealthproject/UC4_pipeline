import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from pydicom import dcmread
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import pathlib
import cv2
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage import color
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut


# Load the scans in given folder path
def load_scan(path):
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    thicknesses = []
    for i in range(len(slices)-1):
        thicknesses.append(np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2]))
    thicknesses = list(set(thicknesses))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[2].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    scan[0].PixelSpacing.append(1.0)
    spacing = np.array(scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = []
    for i in range(len(spacing)):
        resize_factor.append(spacing[i] / new_spacing[i])
    #resize_factor = spacing / new_spacing
    resize_factor.sort(reverse=True)
    new_real_shape = image.shape * np.array(resize_factor)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=400):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces,normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


if __name__ == "__main__":
    root = '/mnt/16c983f0-1f80-4125-846d-2a2f5969cdad/deephealth-uc4/data/raw/dataset_molinette20210418/deephealth_molinette_001_/deephealth_molinette_001_e1/ANON2019/20190923/091938/EX1/SE1'
    first_patient = load_scan(root)
    first_patient_pixels = get_pixels_hu(first_patient)

    pix_resampled, spacing = resample(first_patient_pixels, first_patient, np.array([1, 1, 1]))
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)

    plot_3d(pix_resampled, -500)
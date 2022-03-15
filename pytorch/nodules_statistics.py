from importlib.resources import path
import math
import os
import cc3d
import argparse
import pandas as pd
import numpy as np
import pydicom
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes, mesh_surface_area
from glob import glob

def resample(image, slicethickness, pixelspacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([slicethickness] + list(pixelspacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    print(image.min())
    print(image.max())
    print(new_shape)
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=1)
    print(real_resize_factor)  
    #image[image<0] = 0.
    #image[image>0] = 1.
    print(image.min())
    print(image.max())
    return image

def is_nan(x):
    return (x != x)

def get_3D_exam(root, gt_paths):
    np_gt_masks = np.zeros((len(gt_paths), 512, 512))
    np_gt_masks_augm = np.zeros((len(gt_paths), 512, 512))
    for i, gt_path in enumerate(gt_paths):
        if is_nan(gt_path):
            gt = np.zeros((512, 512)).astype(int)
        else:
            gt = np.asarray(Image.open(os.path.join(root, 'masks', gt_path)).convert('L'), dtype=np.int32)/255
        np_gt_masks[i] = gt

    for i in range((len(gt_paths))):
        if i == 0:
            np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool)).astype(int)
        elif i == len(gt_paths)-1:
            np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i-1].astype(bool)).astype(int)
        else:
            np_gt_masks_augm[i] = (np_gt_masks[i-1] .astype(bool) | np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool)).astype(int)

    return np_gt_masks_augm


def count_nodules_statistics(binned_stats, split, root, df, gt_paths, dicom_paths):
    exam_3D_masks = get_3D_exam(root, gt_paths)

    dcm=pydicom.read_file(os.path.join(root, 'images',dicom_paths[0]))
    pixel_spacing=dcm.PixelSpacing
    slices_thickness=dcm.SliceThickness
    voxel_volume=slices_thickness*pixel_spacing[0]*pixel_spacing[1]

    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    #exam_3D_masks = resample(exam_3D_masks, slicethickness, pixel_spacing)
    labels_out = cc3d.connected_components(exam_3D_masks, connectivity=connectivity)
    N = np.max(labels_out) 
    splitted_path = dicom_paths[0].split("/")
    #print(splitted_path)
    patientID, exam = int(splitted_path[-3].split("_")[1]), int(splitted_path[-2].split("_")[1])
    summary = 'Patient: {}, Exam: {}, Nodules: {}'.format(patientID, exam, N)
    print(summary)
    #plot_3d(summary, exam_3D_masks)
    #plot_3d(summary, exam_3D_predictions)

    for label, image in cc3d.each(labels_out, binary=False, in_place=True):
        #plot_3d(summary, image)
        num_pixels_nodule = np.count_nonzero(image==label)
        volume_nodule = voxel_volume*num_pixels_nodule
        diamater = (6*(volume_nodule/math.pi))**(1. / 3)
        if diamater < 3:
            binned_stats[split][0] += 1
        elif diamater >= 3 and diamater < 10:
            binned_stats[split][1] += 1
        elif diamater >= 10 and diamater < 30:
            binned_stats[split][2] += 1
        else:
            binned_stats[split][3] += 1
        row = [patientID, exam, label, volume_nodule, diamater]
        df_length = len(df)
        df.loc[df_length] = row
    return N
def main(args):
    splits = ["test", "val", "train"]
    binned_stats = {"test": [0, 0, 0, 0], "val": [0,0,0,0], "train": [0,0,0,0]}
    for split in splits:
        print("#"*20 + split + "#"*20)
        df = pd.read_csv(os.path.join(args.data, split, "{}_dataset.csv".format(split)))
        column_names = ["patientID", "exam", "nodule", "volume", "diamater"]
        df_output = pd.DataFrame(columns = column_names)
        exams_images = df.groupby(["patientID", "exam"])["image"].apply(list)
        exams_masks = df.groupby(["patientID", "exam"])["mask"].apply(list)
        tot_num_nodules = 0
        for i, _ in enumerate(exams_masks):
            num_nodules_temp = count_nodules_statistics(binned_stats, split, os.path.join(args.data, split), df_output, exams_masks.iloc[i],  exams_images.iloc[i])
            tot_num_nodules += num_nodules_temp
            print("Total number of nodules (ground truth) in split {}: {}".format(split, tot_num_nodules))
        print(binned_stats)
        df_output.to_csv(os.path.join(args.data, split, "{}_nodules_statistics.csv".format(split)))
        print("Total number of nodules (ground truth) in split {}: {}".format(split, tot_num_nodules))

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Nodules statistics script')
    parser.add_argument('-d', '--data', default='/data/deephealth/deephealth-uc4/data/interim/unitochest/',type=str,
                        help='Path to the dataset')
    args = parser.parse_args()

    main(args)
from importlib.resources import path
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

def plot_3d(summary, image, threshold=1):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image

    verts, faces, _, _ = measure.marching_cubes(p)

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
    plt.title(summary)
    plt.show()

def is_nan(x):
    return (x != x)

def get_3D_exam(gt_paths, predictions_paths ):
    np_gt_masks = np.zeros((len(gt_paths), 512, 512))
    np_gt_masks_augm = np.zeros((len(gt_paths), 512, 512))
    np_predictions_masks = np.zeros((len(gt_paths), 512, 512))
    for i, gt_path in enumerate(gt_paths):
        gt = np.asarray(Image.open(gt_path).convert('L'), dtype=np.int32)/255
        p = np.asarray(Image.open(predictions_paths[i]).convert('L'), dtype=np.int32)/255
        np_gt_masks[i] = gt
        np_predictions_masks[i] = p

    for i in range((len(gt_paths))):
        if i == 0:
            np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool)).astype(int)
        elif i == len(gt_paths)-1:
            np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i-1].astype(bool)).astype(int)
        else:
            np_gt_masks_augm[i] = (np_gt_masks[i-1] .astype(bool) | np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool)).astype(int)

    # for i in range((len(gt_paths))):
    #     if i == 0:
    #         np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool) | np_gt_masks[i+2].astype(bool)).astype(int)
    #     elif i == len(gt_paths)-1:
    #         np_gt_masks_augm[i] = (np_gt_masks[i] .astype(bool) | np_gt_masks[i-1].astype(bool) | np_gt_masks[i-2].astype(bool)).astype(int)
    #     elif i == len(gt_paths)-2:
    #         np_gt_masks_augm[i] = (np_gt_masks[i-2] .astype(bool) | np_gt_masks[i-1] .astype(bool) | np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool)).astype(int)
    #     else:
    #         np_gt_masks_augm[i] = (np_gt_masks[i-2] .astype(bool) | np_gt_masks[i-1] .astype(bool) | np_gt_masks[i] .astype(bool) | np_gt_masks[i+1].astype(bool) | np_gt_masks[i+2].astype(bool)).astype(int)

    return np_gt_masks_augm, np_predictions_masks


def count_nodules_statistics(gt_path, predictions_path):
    #df = pd.read_csv(csv_path)
    #exams_images = df.groupby(["patientID", "exam"])["image"].apply(list)
    #exams_masks = df.groupby(["patientID", "exam"])["mask"].apply(list)
    exam_gt = sorted(glob(os.path.join(gt_path,"*.png")))
    exam_predictions = sorted(glob(os.path.join(predictions_path,"*.png")))
    assert(len(exam_gt) == len(exam_predictions))
    #print(exams_images.iloc[exam_id])
    exam_3D_masks, exam_3D_predictions = get_3D_exam(exam_gt, exam_predictions)
    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    #exam_3D_masks = resample(exam_3D_masks, slicethickness, pixel_spacing)
    labels_out = cc3d.connected_components(exam_3D_masks, connectivity=connectivity)
    N = np.max(labels_out) 
    splitted_path = exam_gt[0].split("/")
    #print(splitted_path)
    patientID, exam = int(splitted_path[-3].split("_")[1]), int(splitted_path[-2].split("_")[1])
    summary = 'Patient: {}, Exam: {}, Nodules: {}'.format(patientID, exam, N)
    print(summary)
    #plot_3d(summary, exam_3D_masks)
    #plot_3d(summary, exam_3D_predictions)
    nodules=[]
    pixels=[]
    data=[]
    TP = 0
    FN = 0
    for label, image in cc3d.each(labels_out, binary=False, in_place=True):
        #print(label)
        #print(image.shape)
        #print('unique pixels: ',np.unique(image), image.shape)
        #print('no of pixels: ',np.count_nonzero(image))
        binary_image = image.copy()
        binary_image[binary_image>0] = 1
        #print('unique pixels: ',np.unique(binary_image), binary_image.shape)
        #print('pred unique pixels: ',np.unique(exam_3D_predictions), exam_3D_predictions.shape)
        if np.sum(binary_image*exam_3D_predictions) > 0:
            TP += 1
        else:
            FN += 1
        #plot_3d(summary, image)
        #if np.count_nonzero(image)>0:
        #    nodules.append(label)
        #    p=np.count_nonzero(image==label)
        #    pixels.append(p)
            #data.append([input_path,pixel_spacing,slices_thickness,N,label,p,volume])
            #data_write_csv('train_3d_nodule.csv',data)
            #data=[]
        #print(nodules,pixels)
    return TP, FN, N
def main(args):
    gt_paths = glob(os.path.join(args.data,'*/*/'))
    predictions_paths = glob(os.path.join(args.predictions,'*/*/'))
    TP = 0
    FN = 0
    tot_num_nodules = 0
    for i, _ in enumerate(gt_paths):
        TP_temp, FN_temp, num_nodules_temp = count_nodules_statistics(gt_paths[i], predictions_paths[i])
        TP += TP_temp
        FN += FN_temp
        tot_num_nodules += num_nodules_temp
        print("TP: {}".format(TP))
        print("FN: {}".format(FN))
        print("Sensitivity: {}".format(TP/(TP+FN)))
        print("Total number of nodules (ground truth): {}".format(tot_num_nodules))
        #pd.to_csv(count_nodules_statistics(csv_paths[split]))
    print("TP: {}".format(TP))
    print("FN: {}".format(FN))
    print("Sensitivity: {}".format(TP/(TP+FN)))
    print("Total number of nodules (ground truth): {}".format(tot_num_nodules))

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Nodules statistics script')
    parser.add_argument('-d', '--data', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/unitochest/test_new/ground_truth',type=str,
                        help='Path to the dataset')
    parser.add_argument('-p', '--predictions', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/unitochest/test_new/predictions',type=str,
                        help='Path to the dataset')
    parser.add_argument('-o', '--output', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/unitochest/test_new/predictions',type=str,
                        help='Output path')
    args = parser.parse_args()

    main(args)
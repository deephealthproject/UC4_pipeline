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

def get_3D_exam(root, images_paths, masks_paths):
    # we assume that PixelSpacing and SliceThickness are equals among slices of the same CT scan
    dcm = pydicom.read_file(os.path.join(root, "images", images_paths[0]))
    pixel_spacing = dcm.PixelSpacing
    slices_thickness = dcm.SliceThickness
    
    np_masks = np.zeros((len(masks_paths), 512, 512))
    for i, masks_path in enumerate(masks_paths):
        if is_nan(masks_path):
            mask = np.zeros((512, 512), dtype=np.int32)
        else:
            mask = np.asarray(Image.open(os.path.join(root, "masks", masks_path)), dtype=np.int32)/255
        np_masks[i] = mask

    return np_masks, slices_thickness, pixel_spacing


def count_nodules_statistics(root, csv_path):
    df = pd.read_csv(csv_path)
    exams_images = df.groupby(["patientID", "exam"])["image"].apply(list)
    exams_masks = df.groupby(["patientID", "exam"])["mask"].apply(list)
    assert(len(exams_images) == len(exams_masks))
    for exam_id in range(len(exams_images)):
        #print(exams_images.iloc[exam_id])
        exam_3D_masks, slicethickness, pixel_spacing = get_3D_exam(root, exams_images.iloc[exam_id], exams_masks.iloc[exam_id])
        volume_voxel = slicethickness*pixel_spacing[0]*pixel_spacing[1]
        connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        #exam_3D_masks = resample(exam_3D_masks, slicethickness, pixel_spacing)
        labels_out = cc3d.connected_components(exam_3D_masks, connectivity=connectivity)
        N = np.max(labels_out) 
        splitted_path = exams_images.iloc[exam_id][0].split("/")
        patientID, exam = int(splitted_path[0].split("_")[1]), int(splitted_path[1].split("_")[1])
        summary = 'Patient: {}, Exam: {}, Nodules: {}'.format(patientID, exam, N)
        print(summary)
        #plot_3d(summary, exam_3D_masks)
        nodules=[]
        pixels=[]
        data=[]
        for label, image in cc3d.each(labels_out, binary=False, in_place=True):
            print(label)
            
            print('unique pixels: ',np.unique(image), image.shape)
            print('no of pixels: ',np.count_nonzero(image) )
            if np.count_nonzero(image)>0:
                nodules.append(label)
                p=np.count_nonzero(image==label)
                pixels.append(p)
                volume=p*volume_voxel
                #data.append([input_path,pixel_spacing,slices_thickness,N,label,p,volume])
                #data_write_csv('train_3d_nodule.csv',data)
                #data=[]
            print(nodules,pixels)

def main(args):
    splits = ["train", "val", "test"]
    csv_paths = {}
    for split in splits:
        csv_paths[split] = os.path.join(args.data, split, "{}_dataset.csv".format(split))
        count_nodules_statistics(os.path.join(args.data, split), csv_paths[split])
        break
        #pd.to_csv(count_nodules_statistics(csv_paths[split]))

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Nodules statistics script')
    parser.add_argument('-d', '--data', default='/data/deephealth/deephealth-uc4/data/interim/unitochest/',type=str,
                        help='Path to the dataset')
    parser.add_argument('-d', '--predictions', default='/data/deephealth/deephealth-uc4/data/interim/unitochest/',type=str,
                        help='Path to the dataset')
    parser.add_argument('-o', '--output', default='/data/deephealth/deephealth-uc4/data/interim/unitochest/',type=str,
                        help='Output path')
    args = parser.parse_args()

    main(args)
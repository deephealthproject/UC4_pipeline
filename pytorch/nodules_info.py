import cc3d
import numpy as np
import os
from PIL import Image
from numpy import asarray
import matplotlib
matplotlib.use('Agg')
import csv
import pydicom

import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes, mesh_surface_area

def process(image):
    return image

def data_write_csv(file_name, datas): # file_name is the path to write the CSV file, datas is the list of data to be written
    with open(file_name, 'a', encoding='UTF8') as f:
         
        writer = csv.writer(f)
        for data in datas:
            writer.writerow(data)
        print("Saved the file successfully, logging is over")


csv_file='train_subfolders.csv'

csv_path = os.path.join(root, "{}_dataset.csv".format(split))
        self.dataframe = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

with open(csv_file) as f:
        len_csv=sum(1 for line in f)
        
print(len_csv)
a=[]
with open(csv_file, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print(row, len(row))
        if(len(row)>0):
                
            input_path=str(row[0])
            print(input_path)


            name=input_path.split('masks')
            patient=name[1]
            pre=name[0]
            pre=pre.split('/mas')
            prefix=pre[0]
            dicom=prefix+'images'+patient
            dicom=dicom.rsplit("/", 1)
            
            dicom_0=dicom[0]
            print('dir: ', dicom_0)
            dicom_path=os.listdir(dicom_0)
            
            print(str(dicom_0)+'/'+str(dicom_path[0]))
            dicom_file=str(dicom_0)+'/'+str(dicom_path[0])
            print('dicom_file: ',dicom_file)
            dicom_img=os.listdir(dicom_file)
            print('first img: ',dicom_img[5])
            dicom_full_path=dicom_file+'/'+dicom_img[5]
            print(dicom_full_path)

            dcm=pydicom.read_file(dicom_full_path)
            pixel_spacing=dcm.PixelSpacing
            slices_thickness=dcm.SliceThickness
            resultant=slices_thickness*pixel_spacing[0]*pixel_spacing[1]


            
            
            



            mask_files = sorted(os.listdir(input_path))
            labels_in = np.ones((512, 512, len(mask_files)), dtype=np.int32)
            for i, mask_file in enumerate(mask_files):
                # load the image
                image = Image.open(os.path.join(input_path,mask_file))
                # convert image to numpy array
                data = asarray(image)/255
                labels_in[:,:,i] = data
            #labels_in = np.ones((512, 512, 512), dtype=np.int32)
            #plot_3d(labels_in, 1)
            #labels_out = cc3d.connected_components(labels_in) # 26-connected

            connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
            labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)
            N = np.max(labels_out) # costs a full read
            print('max: ',N)
            #print('all labels: ', labels_out)
            nodules=[]
            pixels=[]
            data=[]




            for label, image in cc3d.each(labels_out, binary=False, in_place=True):
                print(label)
                
                process(image)
                print('unique pixels: ',np.unique(image), image.shape)
                print('no of pixels: ',np.count_nonzero(image) )
                if np.count_nonzero(image)>0:
                    nodules.append(label)
                    p=np.count_nonzero(image==label)
                    pixels.append(p)
                    volume=p*resultant
                    data.append([input_path,pixel_spacing,slices_thickness,N,label,p,volume])
                    data_write_csv('train_3d_nodule.csv',data)
                    data=[]
                print(nodules,pixels)
                       
#stats = cc3d.statistics(image)
#print('stats: ',len(stats),stats)
#edges = cc3d.region_graph(labels_out, connectivity=connectivity) 

#help(cc3d.statistics)
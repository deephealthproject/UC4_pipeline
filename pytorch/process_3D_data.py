import os
import argparse
import shutil
import pandas as pd 
import numpy as np
from pathlib import Path
from PIL import Image
from utils.dicom_utils import dicom_to_png 
import math 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def is_list_nan(list):
    output = []
    for l in list:
        output.append(type(l) != str and math.isnan(l))
    return np.array(output).all()

def save_chunks(index, input_path, output_path, chunk_size, images_chunks, masks_chunks):
    patientID = images_chunks[0].split("/")[0].split("_")[1]
    exam = images_chunks[0].split("/")[1].split("_")[1]
    if not is_list_nan(masks_chunks) and len(masks_chunks)==chunk_size:
        output_chunk_images = np.zeros((chunk_size, 512, 512))
        output_chunk_masks = np.zeros((chunk_size, 512, 512))
        #print(masks_chunks)
        #print(is_list_nan(masks_chunks))
        for i, image in enumerate(images_chunks):
            #print(images_chunks[i])
            #print(masks_chunks[i])
            im = Image.open(os.path.join(input_path, 'images', images_chunks[i]))
            output_chunk_images[i] = np.array(im.getdata()).reshape((1,512,512))
            if type(masks_chunks[i]) != str and math.isnan(masks_chunks[i]):
               output_chunk_masks[i] = np.zeros((1, 512, 512))
            else:
                msk = Image.open(os.path.join(input_path, 'masks', masks_chunks[i]))
                output_chunk_masks[i] = np.array(msk.getdata()).reshape((1,512,512))
        #print(output_chunk_images.shape)
        #print(output_chunk_masks.shape)
        print(patientID, exam, index)
        Path(os.path.join(output_path, "images", "patient_{}".format(patientID), "exam_{}".format(exam))).mkdir(parents=True, exist_ok=True) 
        Path(os.path.join(output_path, "masks", "patient_{}".format(patientID), "exam_{}".format(exam))).mkdir(parents=True, exist_ok=True) 
        np.save(os.path.join(output_path, "images", "patient_{}".format(patientID), "exam_{}".format(exam), "patient_{}_exam_{}_{}.npy".format(patientID, exam, index)), output_chunk_images)
        np.save(os.path.join(output_path, "masks", "patient_{}".format(patientID), "exam_{}".format(exam),"patient_{}_exam_{}_{}_mask.npy".format(patientID, exam, index)), output_chunk_masks)

def get_chunks(split, input_path, output_path, chunk_size, df_path):
    print(chunk_size)
    df = pd.read_csv(os.path.join(input_path, "{}_dataset.csv".format(split)))
    print(df.head())
    images_gp = df.groupby(['patientID', 'exam']).agg({'image': lambda x: x.tolist()})
    masks_gp = df.groupby(['patientID', 'exam']).agg({'mask': lambda x: x.tolist()})
    images_gp_list = list(images_gp['image'])
    masks_gp_list = list(masks_gp['mask'])
    print(len(images_gp_list))
    for i, l in enumerate(images_gp_list):
        images_chunks = list(chunks(l, chunk_size))
        masks_chunks = list(chunks(masks_gp_list[i], chunk_size))
        for c, chuck in enumerate(images_chunks):
            print(c)
            save_chunks(c, input_path, output_path, chunk_size, images_chunks[c], masks_chunks[c]) 
        #print(list(images_chunks)[0]) #list of lists (chunks)
        #print(list(masks_chunks)[0]) #list of lists (chunks)

def main(args):
    splits = ["train", "val", "test"]
    for split in splits: 
        get_chunks(split, os.path.join(args.input_path, split), os.path.join(args.output_path, split), args.chunk_size, "{}_dataset.csv".format(split))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final dataset')
    parser.add_argument('-d', '--input_path', default='/data/deephealth-uc4/data/processed/unitochest',type=str,
                        help='Dicom dataset path')
    parser.add_argument('-o', '--output_path', default='/data/deephealth-uc4/data/processed/unitochest_3D_5',type=str,
                        help='Dataset output path')
    parser.add_argument('-c', '--chunk_size', default=5, type=int, help='Chunk size of each example')
    args = parser.parse_args()
    main(args)
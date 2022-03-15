import shutil
import logging
import multiprocessing
import os
import argparse
import numpy as np
from scipy import ndimage
from utils.preprocessing import remove_noise
import cv2
import pandas as pd
import utils.file_utils as fu
from pathlib import Path
from joblib import Parallel, delayed
from utils.dicom_utils import dicom_to_png

logging.basicConfig(level=logging.INFO, format='')

def get_exam_id_from_path(path):
    exam_info = path.split('/')
    exam_id = exam_info[1].split('_')[1]
    return exam_id

def copy_masks(exam_paths, output_path, patient_id, exam_id, df):
    fu.ensure_dir(os.path.join(output_path,
                               'masks',
                               'patient_{}'.format(patient_id.zfill(4)),
                               'exam_{}'.format(exam_id)))
    original_files_names = exam_paths[0]
    new_files_names = exam_paths[1]
    for i, old_path in enumerate(original_files_names):
        # new_path = os.path.join(output_path, 'masks', new_files_names[i])
        # column_image = new_files_names[i].replace('_mask', '')
        # column_image = column_image.replace('png', 'npy')
        # df['mask'][df['image'] == column_image] = new_files_names[i]
        # img_mask = cv2.imread(str(old_path), 0)
        # label = ndimage.binary_fill_holes(img_mask).astype(float)
        # cv2.imwrite(new_path, label*255)
        # shutil.copy(old_path, new_path)

        img_mask = cv2.imread(str(old_path), 0)
        img_mask[img_mask>0] = 255
        if np.min(img_mask) == 0 and np.max(img_mask) == 255 and np.unique(img_mask).shape[0] == 2:
            label = ndimage.binary_fill_holes(img_mask).astype(float)
            new_path = os.path.join(output_path, 'masks', new_files_names[i])
            column_image = new_files_names[i].replace('_mask', '')
            column_image = column_image.replace('png', 'dcm')
            #print(column_image)
            #print(new_files_names[i])
            df['mask'][df['image'] == column_image] = new_files_names[i]
            cv2.imwrite(new_path, label*255)
        else:
            print(str(old_path))

def rename_raw_dicoms(exam_paths, output_path, patient_id, exam_id, df):
    fu.ensure_dir(os.path.join(output_path,
                               'images',
                               'patient_{}'.format(patient_id.zfill(4)),
                               'exam_{}'.format(exam_id)))
    original_files_names = exam_paths[0]
    new_files_names = exam_paths[1]
    for i, old_path in enumerate(original_files_names):
        #logging.info('Converting image {}'.format(old_path))
        new_path = os.path.join(output_path, 'images', new_files_names[i])
        #patient_001/exam_1/001_1_84.dcm
        slice_id = int(new_files_names[i].split("/")[2].split("_")[2].split(".")[0])
        df.loc[len(df)] = [int(patient_id), int(exam_id), int(slice_id), new_files_names[i], '']
        shutil.copy(old_path, new_path)
        #logging.info('Finished converting image {}.'.format(old_path))

def rename_raw_patient(patient_id, patient_files, output_path, df):
    #logging.info('Converting raw files for patient {}...'.format(patient_id))
    images = patient_files['images']
    masks = patient_files['masks']

    #logging.info('Converting images...')
    for exam_paths in images:
        exam_id = get_exam_id_from_path(exam_paths[1][0])
        rename_raw_dicoms(exam_paths, output_path, patient_id, exam_id, df)
    #logging.info('Finished converting images.')

    #logging.info('Coping masks...')
    # Note that no every image has a mask so len(images) != len(masks)
    for exam_paths in masks: 
        exam_id = get_exam_id_from_path(exam_paths[1][0])
        copy_masks(exam_paths, output_path, patient_id, exam_id, df)
    #logging.info('Finished coping masks.')
    #logging.info('Finished converting raw files for patient {}...'.format(patient_id))

def rename_raw_dataset(output_path, dataset_files, df):
    for patient_id, patient_files in dataset_files.items():
        rename_raw_patient(patient_id, patient_files, output_path, df)

def get_dataset_csv(path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=['patientID', 'exam', 'slice', 'image', 'mask'])
    return df

def main(args, parallel=False):
    csv_path = os.path.join(args.output_path, 'dataset.csv')
    logging.info('Reading raw files...')
    dataset_files = fu.get_dataset_files(args.dir_path)
    df = get_dataset_csv(csv_path)
    logging.info('Refactoring started!')

    if parallel:
        # TO FIX: This does not write anything on dataframe!!
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(rename_raw_patient)(patient_id, patient_files, args.output_path, df) for patient_id, patient_files in dataset_files.items())
    else:
        rename_raw_dataset(args.output_path, dataset_files, df)
    
    df = df.sort_values(by=['patientID', 'exam', 'slice'], ascending=[True, True, True])
    df.to_csv(csv_path, index=False)
    logging.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DICOM to png')
    parser.add_argument('-d', '--dir_path', default='/data/deephealth-uc4/data/raw/dataset_molinette20210418',type=str,
                        help='Raw dataset path')
    parser.add_argument('-o', '--output_path', default='/data/deephealth-uc4/data/raw/unitochest2',type=str,
                        help='Dataset output path')
    args = parser.parse_args()
    main(args, parallel=False)
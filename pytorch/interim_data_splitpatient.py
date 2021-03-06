import os
import shutil
import shutil
import argparse
import pandas as pd
import numpy as np
import utils.file_utils as fu
from tqdm import tqdm
from sklearn.model_selection import train_test_split

tqdm.pandas()
np.random.seed(0)

def get_flat_list(l):
    return [sublist for sublist in l]

def split_train_test2(df, train_size=0.9, sort=False):
    train_list = []
    test_list = []
    train_list_patiendID = []
    train_num_images = 0
    tot_images = len(df)
    max_tot_images = int(tot_images * train_size)
    df = df.sample(frac=1, random_state=0)
    grouped = df.groupby('patientID', sort=False, as_index=False)
    grouped_count = grouped.count()
    if sort:
        grouped_count = grouped_count.sort_values(by=['image'])
    #print(list(grouped_count.iterrows()))
    for index, row in grouped_count.iterrows():
        #print(patientID)
        patientID = row['patientID']
        if train_num_images >= max_tot_images:
            test_list.append(df[df['patientID'] == patientID])
            #print('test')
        elif train_num_images + row['image'] > max_tot_images:
            #print('here')
            patient_with_min_imgs = grouped_count[grouped_count['image'] == grouped_count['image'].min()].iloc[0]
            train_list.append(patient_with_min_imgs)
            train_list_patiendID.append(patient_with_min_imgs['patientID'])
            train_num_images += patient_with_min_imgs['image']
            test_list.append(df[df['patientID'] == patientID])
        elif patientID not in train_list_patiendID:
            #print('there')
            #print(patientID)
            #print(df[df['patientID'] == patientID])
            train_list.append(df[df['patientID'] == patientID])
            train_num_images += row['image']

    df_train = pd.DataFrame(pd.concat(train_list), columns = df.columns)
    df_train = df_train.dropna(subset=['patientID'])
    df_test = pd.DataFrame(pd.concat(test_list),  columns = df.columns)
    df_test = df_test.dropna(subset=['patientID'])
    df_train['patientID'] = df_train['patientID'].astype('int64')
    df_test['patientID'] = df_test['patientID'].astype('int64')

    assert(set(df_train.patientID.values).isdisjoint(set(df_test.patientID.values)))
    return df_train, df_test

def split_train_test(df, train_size=0.9):
    np.random.seed(0)
    msk = np.random.rand(len(df)) < train_size
    df_train = df[msk]
    df_test = df[~msk]
    return df_train, df_test

def copy_element(x, input_path, output_path):
    old_image_path = os.path.join(input_path, 'images', x['image'])
    old_image_path_unrolled = os.path.join(input_path, 'images', x['image'].split("/")[-1])
    new_image_path = old_image_path_unrolled.replace(input_path, output_path)
    fu.ensure_dir(os.path.dirname(new_image_path))
    shutil.copy(old_image_path, new_image_path)

    if type(x['mask']) == float and np.isnan(x['mask']):
        a = 3
    else:
        old_mask_path = os.path.join(input_path, 'masks', x['mask'])
        old_mask_path_unrolled = os.path.join(input_path, 'masks', x['mask'].split("/")[-1])
        new_mask_path = old_mask_path_unrolled.replace(input_path, output_path)
        new_mask_path = new_mask_path.replace('masks', 'ground_truth')
        fu.ensure_dir(os.path.dirname(new_mask_path))
        shutil.copy(old_mask_path, new_mask_path)

def copy_dataset(df, input_path, output_path):
    df.progress_apply(lambda x: copy_element(x, input_path, output_path), axis=1)

# def create_df(image_paths, masks_paths):
#     out = []
#     for path in image_paths:
#         filename = os.path.basename(path)
#         filename_splitted = filename.split("_")
#         patient_id = int(filename_splitted[0])
#         exam = filename_splitted[1]
#         slice_num = filename_splitted[2].split(".")[0]
#         out.append([patient_id, exam, slice_num, path, None])
#     df = pd.DataFrame(out, columns=["patientID", "exam", "slice", "image", "mask"])
#     df = df.reset_index().set_index(["patientID", "exam", "slice"])
#     print(df.head())
#     for path in masks_paths:
#         filename = os.path.basename(path)
#         filename_splitted = filename.split("_")
#         patient_id = int(filename_splitted[0])
#         exam = filename_splitted[1]
#         slice_num = filename_splitted[2].split(".")[0]
#         #print(path)
#         #print(patient_id, exam, slice_num)
#         df.loc[[patient_id, exam, slice_num]]["masks"] = path
#         #df.loc[(df["patientID"] == patient_id) & (df["exam"] == exam) & (df["slice"] == slice_num), "mask"] = path
#     return df

def main(args):
    csv_path = os.path.join(args.input_path, 'dataset.csv')
    shutil.rmtree(args.output_path, ignore_errors = True)
    df = pd.read_csv(csv_path)
    print(df.head())
    print("Total number of slices: {}".format(len(df)))
    print(("Number of patients: {}".format(df.patientID.nunique())))
    df_mask = df.drop(df.query('mask.isnull().values').sample(frac=1, random_state=0).index)
    print(len(df_mask))
    print("Total number of slices with masks: {}".format(len(df)))
    print(("Number of patients with masks: {}".format(df.patientID.nunique())))
    # root = args.input_path
    #image_dir = os.path.join(root, 'images')#, split)
    # label_dir = os.path.join(root, 'masks')#, split)
    #images_paths = sorted([path.relative_to(os.path.join(args.input_path, 'images')) for path in Path(image_dir).rglob('*.npy')])
    #print(len(images_paths))
    # masks_paths = sorted([path.relative_to(os.path.join(args.input_path, 'masks')) for path in Path(label_dir).rglob('*.png')])
    # df = create_df(images_paths, masks_paths)
    # print(df.head())
    #csv_path = os.path.join(args.output_path, 'dataset.csv')
    #logging.info('Reading raw files...')
    #dataset_files = fu.get_dataset_files(args.input_path)
    #df = fu.get_dataset_csv(csv_path) 
    #print(df.head())

    patients_id = list(df.patientID.unique())
    train_patients_id, val_patients_id = train_test_split(patients_id, test_size=0.2, random_state=42)
    val_patients_id, test_patients_id = train_test_split(val_patients_id, test_size=0.5, random_state=42)
    df_train = df[df["patientID"].isin(train_patients_id)]
    df_val = df[df["patientID"].isin(val_patients_id)]
    df_test = df[df["patientID"].isin(test_patients_id)]
    
    print("Number of slices in training set: {}".format(len(df_train)))
    print(("Number of patients in training set: {}".format(df_train.patientID.nunique())))
    print("Number of slices in validation set: {}".format(len(df_val)))
    print(("Number of patients in validation set: {}".format(df_val.patientID.nunique())))
    print("Number of slices in test set: {}".format(len(df_test)))
    print(("Number of patients in test set: {}".format(df_test.patientID.nunique())))

    #print(set(df.patientID.unique()))
    set_train= set(df_train.patientID.unique())
    set_val = set(df_val.patientID.unique())
    set_test = set(df_test.patientID.unique())
    set_valtest = set_val|set_test
    set_trainvaltest = set_valtest|set_train
    print(len(set_valtest))
    print(set(df.patientID.unique())-set_trainvaltest)

    df_train_f = df[df['patientID'].isin(list(df_train['patientID'].unique()))]
    df_val_f = df[df['patientID'].isin(list(df_val['patientID'].unique()))]
    df_test_f = df[df['patientID'].isin(list(df_test['patientID'].unique()))]
    copy_dataset(df_train_f, args.input_path, os.path.join(args.output_path, 'training'))
    copy_dataset(df_val_f, args.input_path, os.path.join(args.output_path, 'validation'))
    copy_dataset(df_test_f, args.input_path, os.path.join(args.output_path, 'test'))

    df_train_f["image"] = df_train_f["image"].str.split("/").str[-1]
    df_train_f["mask"] = df_train_f["mask"].str.split("/").str[-1]

    df_val_f["image"] = df_val_f["image"].str.split("/").str[-1]
    df_val_f["mask"] = df_val_f["mask"].str.split("/").str[-1]

    df_test_f["image"] = df_test_f["image"].str.split("/").str[-1]
    df_test_f["mask"] = df_test_f["mask"].str.split("/").str[-1]

    df_train_f.to_csv(os.path.join(args.output_path, 'training', 'training_dataset.csv'), index=False)
    df_val_f.to_csv(os.path.join(args.output_path, 'validation', 'validation_dataset.csv'), index=False)
    df_test_f.to_csv(os.path.join(args.output_path, 'test', 'test_dataset.csv'), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final dataset')
    parser.add_argument('-d', '--input_path', default='/data/deephealth-uc4/data/raw/unitochest',type=str,
                        help='Raw dataset path')
    parser.add_argument('-o', '--output_path', default='/data/deephealth-uc4/data/interim/unitochest_eddl',type=str,
                        help='Dataset output path')
    args = parser.parse_args()
    main(args)

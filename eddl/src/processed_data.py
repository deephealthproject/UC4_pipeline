
# ATTENTION!
# USE FROM dhealth/pylibs-toolkit:0.10.0-cudnn TO CREATE THE YAML FILE!!!!
import os
import argparse
import shutil
import pandas as pd 
import numpy as np
import pyecvl.ecvl as ecvl
from PIL import Image
from pathlib import Path
from utils.dicom_utils import dicom_to_png 

def convert_dicoms(split, input_path, output_path, df_path):
    black_mask = np.zeros((512,512), np.uint8)
    black_mask = Image.fromarray(black_mask)

    df = pd.read_csv(os.path.join(input_path, df_path))
    if split in ["training", "validation"]:
        df = df.drop(df.query('mask.isnull().values').sample(frac=0.98, random_state=0).index)
    #df = df.drop(df.query('mask.isnull().values').sample(frac=1., random_state=0).index)
    # Copy masks
    shutil.copytree(os.path.join(input_path, 'ground_truth'),os.path.join(output_path, 'ground_truth'))
    # Dicom to png
    paths = list(df["image"])
    for path in paths:
        full_path = os.path.join(input_path, "images", path)
        new_path = str(full_path).replace(input_path, output_path)
        new_path = new_path.replace(".dcm",".png")
        Path(new_path).parent.mkdir(parents=True, exist_ok=True)
        image_noextension = path.split(".")[0]
        print("{}_mask.png".format(image_noextension))
        if not "{}_mask.png".format(image_noextension) in df["mask"].tolist():
            print("not present")
            print(df["mask"].iloc[0])
            black_mask.save(os.path.join(output_path, 'ground_truth', "{}_mask.png".format(image_noextension)))
        else:
            print("present")
        dicom_to_png(full_path, new_path, windowing=True, segment=False)

    #df["image"] = df["image"].str.replace('.dcm','.png')
    #df.to_csv(os.path.join(output_path, df_path), index=False)

def main(args):
    splits = ["validation", "test", "training"]
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    for split in splits: 
        convert_dicoms(split, os.path.join(args.input_path, split), os.path.join(args.output_path, split), "{}_dataset.csv".format(split))
    
    # Segmentation dataset for eddl format
    # Possible ground truth suffix or extension if different from images
    suffix = "_mask.png"
    # Possible ground truth name for images that have the same ground truth
    gsd = ecvl.GenerateSegmentationDataset(args.output_path, suffix)
    seg_d = gsd.GetDataset()
    print("dumping segmentation dataset")
    seg_d.Dump(os.path.join(args.output_path, "dataset_molinette.yml"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final dataset')
    parser.add_argument('-d', '--input_path', default='/data/deephealth-uc4/data/interim/unitochest',type=str,
                        help='Dicom dataset path')
    parser.add_argument('-o', '--output_path', default='/data/deephealth-uc4/data/processed/unitochest',type=str,
                        help='Dataset output path')
    args = parser.parse_args()
    main(args)

    #find /data/deephealth/deephealth-uc4/data/processed/unitochest/validation/ground_truth/ -type f | wc -l
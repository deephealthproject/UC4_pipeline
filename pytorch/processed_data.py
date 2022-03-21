import os
import argparse
import shutil
import pandas as pd 
from pathlib import Path
from utils.dicom_utils import dicom_to_png 

def convert_dicoms(input_path, output_path, df_path):
    df = pd.read_csv(os.path.join(input_path, df_path))
    df["image"] = df["image"].str.replace('.dcm','.png')

    # Copy masks
    shutil.copytree(os.path.join(input_path, 'ground_truth'),os.path.join(output_path, 'ground_truth'))
    # Dicom to png
    for path in Path(input_path).rglob('*.dcm'):
        new_path = str(path).replace(input_path, output_path)
        new_path = new_path.replace(".dcm",".png")
        Path(new_path).parent.mkdir(parents=True, exist_ok=True)
        dicom_to_png(path, new_path, windowing=True, segment=False)

    df.to_csv(os.path.join(output_path, df_path), index=False)

def main(args):
    splits = ["training", "validation", "test"]

    for split in splits: 
        convert_dicoms(os.path.join(args.input_path, split), os.path.join(args.output_path, split), "{}_dataset.csv".format(split))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final dataset')
    parser.add_argument('-d', '--input_path', default='/data/deephealth-uc4/data/interim/unitochest',type=str,
                        help='Dicom dataset path')
    parser.add_argument('-o', '--output_path', default='/data/deephealth-uc4/data/processed/unitochest',type=str,
                        help='Dataset output path')
    args = parser.parse_args()
    main(args)
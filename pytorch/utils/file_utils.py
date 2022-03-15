import os
import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def get_files_names(patient_dir, patient_id, exam_id, is_mask):
    original_files_names = []
    new_files_names = []
    for path, subdirs, files in os.walk(patient_dir):
        for name in files:
            a = os.path.join(path, name)
            original_files_names.append(os.path.join(path, name))
            if is_mask:
                slice_id = name.split('_')[4]
                new_name = 'patient_{}/exam_{}/{}_{}_{}_mask.png'.format(patient_id.zfill(4), exam_id, patient_id.zfill(4), exam_id, slice_id.zfill(4))
            else:
                slice_id = name.replace('IMG', '') 
                new_name = 'patient_{}/exam_{}/{}_{}_{}.dcm'.format(patient_id.zfill(4), exam_id, patient_id.zfill(4), exam_id, slice_id.zfill(4))
            new_files_names.append(new_name)
    return original_files_names, new_files_names

def get_patient_files(patient_dir, patient_id):
    patient_dirs = [os.path.join(patient_dir, dir) for dir in sorted(os.listdir(patient_dir))]
    masks = []
    images = []
    for i, dir in enumerate(patient_dirs):
        dir_path = Path(dir).name
        if 'MASCHERE' in dir or 'MASCHERA' in dir or 'masks' in dir:
            exam_id = dir_path.split('_')[3][1:]
            masks.append(get_files_names(dir, patient_id, exam_id, True))
        else:
            exam_id = dir.split('/')[-1].split("_")[3][1:]
            images.append(get_files_names(dir, patient_id, exam_id, False))
    return images, masks

def get_dataset_files(dir_path):
    patients_dirs = [dir for dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir))]
    patients_dirs.sort()
    patients_ids = [patients_dir.split('_')[2] for patients_dir in patients_dirs]
    patients_dirs = list(map(lambda dir: os.path.join(dir_path, dir), patients_dirs))
    patients_files = {}
    for i, patients_id  in enumerate(patients_ids):
        patients_files[patients_id] = {}
        images, masks = get_patient_files(patients_dirs[i], patients_id)
        patients_files[patients_id]['images'] = images
        patients_files[patients_id]['masks'] = masks
    return patients_files

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
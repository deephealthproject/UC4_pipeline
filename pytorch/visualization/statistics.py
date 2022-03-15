import os
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

dirName = "data/raw/dataset_molinette20210418"
machine = {}
patients = {}
width = 1.0    
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for input_path in listOfFiles:
    patient_id = input_path.split("/")[3].split("_")[2]
    if '.png' not in input_path:
        dcm = pydicom.dcmread(str(input_path))
        if dcm.PhotometricInterpretation != "MONOCHROME2":
            print(dcm.PhotometricInterpretation)
        patients[patient_id] = dcm.StationName
        if dcm.StationName in machine.keys():
            machine[dcm.StationName] += 1
        else:
            machine[dcm.StationName] = 1
df = pd.DataFrame.from_dict(patients, orient="index")
df.to_csv("patients_stationnames.csv")
plt.bar(machine.keys(), machine.values(), width, color='g')
plt.show()



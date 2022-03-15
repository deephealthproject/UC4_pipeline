# Convert to PNG via PIL
# https://github.com/python-pillow/Pillow
import os
import pydicom
from PIL import Image
import numpy as np
#import torch
#from torchvision import transforms
#import torchvision
import datetime
import cv2

from pathlib import Path
from pydicom.dataset import Dataset, FileDataset

BASE_DIR = Path(__file__).resolve().parents[2]

base = list(np.arange(0,524288, 2))
adv = list(np.arange(1,524288, 2))

def win_scale(data, wl, ww, dtype, out_range):
	"""
	Scale pixel intensity data using specified window level, width, and intensity range.
	"""

	data_new = np.empty(data.shape, dtype=np.double)
	data_new.fill(out_range[1]-1)

	data_new[data <= (wl-ww/2.0)] = out_range[0]
	data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
	     ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
	data_new[data > (wl+ww/2.0)] = out_range[1]-1
	return data_new.astype(dtype)

def ct_win(im, wl, ww, dtype, out_range):
	"""
	Scale CT image represented as a `pydicom.dataset.FileDataset` instance.
	"""

	# Convert pixel data from Houndsfield units to intensity:
	intercept = int(im[(0x0028, 0x1052)].value)
	slope = int(im[(0x0028, 0x1053)].value)
	data = (slope*im.pixel_array+intercept)

	# Scale intensity:
	return data#win_scale(data, wl, ww, dtype, out_range)

def dicomtouint8(data):
	rows = int(data[(0x0028,0x0010)].value)
	cols = int(data[(0x0028,0x0011)].value)
	raw_full = data.PixelData
	print(raw_full.shape)
	raw = [raw_full[x] for x in base]
	converted = ((np.asarray(map(ord, raw))).reshape(rows, cols))
	adv_raw = [raw_full[x] for x in adv]
	converted += 256*((np.asarray(map(ord, adv_raw))).reshape(rows,cols))
	converted = converted.astype(np.double)
	intercept = float(data[(0x0028, 0x1052)].value)
	slope = float(data[(0x0028, 0x1053)].value)
	converted = (slope*converted+intercept)
	converted = win_scale(converted, data[(0x0028, 0x1050)].value, data[(0x0028, 0x1051)].value, np.uint8, [0, 255])
	return converted

def tensortodicom(inputdir, tensor, outname):
	ds = pydicom.dcmread(inputdir)

	 # the PNG file to be replace
	# (8-bit pixels, black and white)
	np_frame = np.array((tensor*100)/1,dtype=np.uint16)
	#np_frame = np.array(inputpng*6362 - 97,dtype=np.uint16)
	#print(np.min(np_frame), np.max(np_frame))
	#ds.Rows = 512#im_frame.height
	#ds.Columns = #im_frame.width
	#ds.PhotometricInterpretation = "MONOCHROME2"
	ds.PatientName = (outname.split('/'))[-1][:-4]
	ds.SamplesPerPixel = 1
	ds.BitsStored = 16
	ds.BitsAllocated = 16
	ds.HighBit = 15
	ds.PixelRepresentation = 0
	ds.RescaleIntercept = 0#np.min(np_frame)
	ds.RescaleSlope = 0.01#0.01
	ds.SmallestImagePixelValue = np.min(np_frame)
	ds.LargestImagePixelValue = np.max(np_frame)
	ds.WindowCenter = 128.0
	ds.WindowWidth = 256.0
	#ds.PixelPaddingValue = 256
	ds.PixelData = np_frame.tobytes()
	pydicom.filewriter.dcmwrite(outname, ds)

def pngtodicom(inputdir, inputpng, outname, network):
	ds = pydicom.dcmread(inputdir)

	im_frame = Image.open(outname.replace('.dcm', '.png')).convert('L') # the PNG file to be replace
	# (8-bit pixels, black and white)
	np_frame = np.array(im_frame.getdata(),dtype=np.uint8)
	ds.PhotometricInterpretation = "MONOCHROME2"
	ds.DerivationDescription = "EIDOSLAB UNITO - Inference with NN#2/"+str(network)
	ds.PatientName = (outname.split('/'))[-1][:-4]
	ds.SamplesPerPixel = 1
	ds.BitsStored = 8
	ds.BitsAllocated = 8
	ds.HighBit = 7
	ds.PixelRepresentation = 0
	ds.RescaleIntercept = 0#np.min(np_frame)
	ds.RescaleSlope = 1#0.01
	ds.SmallestImagePixelValue = np.min(np_frame)
	ds.LargestImagePixelValue = np.max(np_frame)
	ds.WindowCenter = 128.0
	ds.WindowWidth = 256.0
	ds.PixelPaddingValue = 256
	ds.PixelData = np_frame.tobytes()
	pydicom.filewriter.dcmwrite(outname, ds)

def handler_graydicomtoheatmapdicom(save_BW_dicom, save_COLOR_dicom):
	dicomtopng(save_BW_dicom)
	grayscaletoheatmap(save_BW_dicom.replace(".dcm", ".png"), save_COLOR_dicom.replace(".dcm", ".png"))
	heatmaptodicom(save_BW_dicom, save_COLOR_dicom)

def handler_graypngtoheatmapdicom(save_BW_dicom, save_COLOR_dicom):
	grayscaletoheatmap(save_BW_dicom.replace(".dcm", ".png"), save_COLOR_dicom.replace(".dcm", ".png"))
	heatmaptodicom(save_BW_dicom, save_COLOR_dicom)

def dicomtopng(map_name):
	map_data = pydicom.dcmread(map_name)
	my_array = (dicomtouint8(map_data)).astype(np.uint8)
	im = Image.fromarray(my_array, 'L')
	save_BW = map_name.replace('.dcm','.png')
	im.save(save_BW)

def grayscaletoheatmap(save_BW, save_COLOR):
	img = cv2.imread(save_BW, 1)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
	cv2.imwrite(save_COLOR,heatmap_img)

def heatmaptodicom(inputdir, outname):
	ds = pydicom.dcmread(inputdir)

	im_frame = Image.open(outname.replace('.dcm', '.png')).convert('RGB') # the PNG file to be replace
	# (8-bit pixels, black and white)
	np_frame = np.array(im_frame.getdata(),dtype=np.uint8)
	ds.PatientName = (outname.split('/'))[-1][:-4]
	ds.PhotometricInterpretation = "RGB"
	ds.SamplesPerPixel = 1
	ds.BitsStored = 8
	ds.BitsAllocated = 8
	ds.HighBit = 7
	ds.PixelRepresentation = 0
	ds.RescaleIntercept = 0#np.min(np_frame)
	ds.RescaleSlope = 1#0.01
	ds.SmallestImagePixelValue = np.min(np_frame)
	ds.LargestImagePixelValue = np.max(np_frame)
	ds.WindowCenter = 128.0
	ds.WindowWidth = 256.0
	ds.PixelPaddingValue = 256
	ds.PixelData = np_frame.tobytes()
	pydicom.filewriter.dcmwrite(outname, ds)

if __name__ == '__main__':
	project_dir = Path(__file__).resolve().parents[2]
	path = BASE_DIR.joinpath('data/raw/IMG168')
	dicomtopng(str(path))
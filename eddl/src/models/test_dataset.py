import cv2 as cv
import os
import glob

images_path = glob.glob('../../data/processed/dataset_molinette/training/images/*.png')

print(images_path)

for image_path in images_path:
    img = cv.imread(image_path)
    print(img.shape)
    print(img.dtype)
    comparison = img[:,:,0] == img[:,:,1]
    equal_arrays = comparison.all()
    print(equal_arrays)

img = cv.imread("../../data/processed/dataset_molinette/test/images/001_1_168.png")
print(img.shape)
print(img.dtype)
print(img[:,:,0].min())
print(img[:,:,0].max())

img = cv.imread("../../data/processed/dataset_molinette/test/ground_truth/001_1_168_mask.png")
print(img.shape)
print(img.dtype)
print(img[:,:,0].min())
print(img[:,:,0].max())
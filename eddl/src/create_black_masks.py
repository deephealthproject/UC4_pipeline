import glob
import os
from PIL import Image
import numpy as np

images_path = "/home/riccardo/projects/uc4/dataset/unitochest_eddl_fulltest/test/images"
gt_path = "/home/riccardo/projects/uc4/dataset/unitochest_eddl_fulltest/test/ground_truth"
images = sorted([os.path.basename(x) for x in glob.glob(os.path.join(images_path, "*.png"))])
masks = sorted([os.path.basename(x) for x in glob.glob(os.path.join(gt_path, "*.png"))])
print(len(masks))
print(len(images))

black_mask = np.zeros((512,512), np.uint8)
black_mask = Image.fromarray(black_mask)
for i, image in enumerate(images):
    image_noextension = image.split(".")[0]
    if not "{}_mask.png".format(image_noextension) in masks:
        black_mask.save(os.path.join(gt_path, "{}_mask.png".format(image_noextension)))

# print(len(images))
# print(len(masks))

# print(images[:10])
# print(masks[:10])
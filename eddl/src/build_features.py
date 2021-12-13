# Copyright (c) 2019-2021 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
DeepHealth dataset generation.
"""

import argparse
import os
import sys

import pyecvl.ecvl as ecvl


def main(args):
    # Segmentation dataset
    # Possible ground truth suffix or extension if different from images
    suffix = "_mask.png"
    # Possible ground truth name for images that have the same ground truth
    gsd = ecvl.GenerateSegmentationDataset(args.seg_dir, suffix)
    seg_d = gsd.GetDataset()
    print("dumping segmentation dataset")
    seg_d.Dump(os.path.join(args.seg_dir, "dataset_molinette.yml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg_dir", metavar="SEGMENTATION_INPUT_DIR")
    main(parser.parse_args(sys.argv[1:]))
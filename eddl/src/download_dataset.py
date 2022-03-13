import gdown
import zipfile

# a file
url = "https://drive.google.com/uc?id=1xmphlKvAD-wbLY1La72YlEG1wWyaDIy0"
output = "/data/unitochest_eddl_fulltest.zip"
output_folder = "/data/"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(output_folder)
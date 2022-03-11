import gdown
import zipfile

# a file
url = "https://drive.google.com/uc?id=1VY3ZeBlQH4sqHt_QiAfxJ9EMkgJLZCmc"
output = "/data/unitochest_eddl.zip"
output_folder = "/data/"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(output_folder)
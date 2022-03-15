FROM dhealth/pylibs-toolkit:1.2.0-cudnn
RUN apt-get update
RUN apt-get install unzip
RUN pip install gdown pandas wandb zipfile38 numpy scipy opencv-python tqdm scikit-learn pydicom pypng pillow
RUN mkdir UC4_pipeline
COPY . UC4_pipeline/


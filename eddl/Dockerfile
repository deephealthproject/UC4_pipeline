FROM dhealth/pylibs-toolkit:1.0.0-cudnn
RUN apt-get update
RUN apt-get install unzip
RUN pip install gdown pandas wandb zipfile38
COPY src deephealth-uc4-pyecvl/


FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set a working directory for the app
WORKDIR /arn/

# Because imports break with versions of pytroch_lightning and unneeded.
RUN pip uninstall --yes torchtext

# Deps for opencv-python, and necessary ARG/ENV for docker build
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y python3-opencv

# Install Prijatelj's public fork of `vast` for the Extreme Value
# Machine and FINCH with recurse-submodules, and get pyflann as dep.
RUN git clone https://github.com/primetang/pyflann.git

# Have to 2to3 the pyflann code...
RUN pip install 2to3==1.0
RUN 2to3 pyflann/
RUN pip install -e pyflann/

RUN git clone --recurse-submodules https://github.com/prijatelj/vast
RUN pip install -e vast/

# Ensuring opencv is headless, may have been changed by above stuffs.
#RUN pip install opencv-python-headless==4.5.5.64
# Install main project and docstr for CLI
COPY . .
RUN pip install -e . -r requirements/arn.txt #dev.txt
RUN pip install docstr==0.0.3rc2

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt update
RUN apt-get update
RUN apt install nano -y
RUN apt install git -y
RUN pip install tensorboard
RUN pip install scipy
RUN pip install efficientnet-pytorch
RUN pip install git+https://github.com/VLL-HD/FrEIA.git
RUN pip install pytorch3d
RUN apt install zip -y
COPY ./*.py /workspace/

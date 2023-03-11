FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt update
RUN apt-get update
RUN apt install nano -y
RUN apt install git -y
RUN apt install wget -y
RUN pip install tensorboard
RUN pip install scipy
RUN pip install efficientnet-pytorch
RUN pip install git+https://github.com/VLL-HD/FrEIA.git
RUN pip install pytorch3d
RUN apt install zip -y
RUN pip install pyyaml
RUN pip install open3d
RUN pip install tqdm
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install kapture
COPY ./*.py /workspace/
COPY ./*.sh /workspace/

RUN useradd -m -d /home/user -u 1005 user
USER 1005:1005

ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

RUN chown -R user:user /workspace/
USER 1005:1005

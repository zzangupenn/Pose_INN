# Pose_INN: Visual Pose Regression and Localization with Invertible Neural Networks
## Supplementary Material Readme

Thank you for reviewing my paper. 

In this supplementary material we provide a guidance on reproducing our work. Our data sampling and training process is fast. The data preparation part takes about 1 hour. For the Pose_INN training, you can get a good result for 5-6 hours. Due to the file size limit, we can't share the custom data with you. This guide will about how to train on the Cambridge and 7scene dataset.

There are four stages in reproducing our work: 
1. Data processing
2. NeRF training
3. NeRF rendering
4. Pose_INN training

## Data processing

1. Download the [Cambridge](https://www.repository.cam.ac.uk/handle/1810/251342) or the [7scene](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset. Extract in directory `[data_dir]/[scene]`. e.g. If your downloaded ShopFacade, your `[data_dir]/[scene]` is `[data_dir]/ShopFacade`. If your are testing on 7scene, extract all the seq zip files in `[data_dir]/[scene]`.

2. We use [kapture](https://github.com/naver/kapture) to convert the dataset format to COLMAP format. A version we used is attached, but it can be install with:

    `pip3 install kapture`

    Copy the `kapture_cambridge.sh` or `kapture_7scene.sh` into `[data_dir]/[scene]` and run it.

3. Modify the first two lines in `data_processing_cambridge_for_nerf.py` or `data_processing_7scene_for_nerf.py` and run it. 

It will create a directory `[data_dir]/[scene]/images`, a json file `[data_dir]/[scene]/transforms.json`, and a npz file `[data_dir]/[scene]/[scene]_H_matrixes.npz`.

## NeRF training

1. We uses [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/v0.1.16) for training the NeRF. Notice the commands provided here may only work with v0.1.16. We attached the version we use. We used their Dockerfile to create a docker container. You can also refer to their wonderful [documentation](https://github.com/nerfstudio-project/nerfstudio/blob/v0.1.16/docs/quickstart/installation.md).

2. Your can use [their docker image](https://hub.docker.com/layers/dromni/nerfstudio/0.1.16/images/sha256-de540fc3e53b62428a4787de78e09feffc84cfbadcca6b4afe4df40a78d3fd92?context=explore):
    
    ```
    docker pull dromni/nerfstudio:0.1.16
    docker run --gpus all --network host \
        -v [data_dir]/:/workspace/ \
        -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it dromni/nerfstudio:0.1.16
    ```

    Or build and run a local docker image:
    ```
    cd nerfstudio
    docker build --tag nerfstudio -f Dockerfile .
    docker run --gpus all --network host \
        -v [data_dir]/[scene]/:/workspace/ \
        -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it nerfstudio
    ```

4. Inside the container, start the NeRF training:
    ```
    ns-train nerfacto --data . --max-num-iterations 10000 nerfstudio-data --train-split-percentage 1  
    ```
    You should be able to see the training progress with their. 
    
    One thing we noticed is that the training is not stable sometimes. If it crashes or the viewer only renders black, you need to retrain the model.  

5. After the training, you should see a `outputs/nerfacto/` folder in your `[data_dir]/[scene]` directory with your trained sessions. 

    We will refer that as the `[session]` folder. e.g. A `[session]` folder may look like `2023-03-11_062711`.

    In the nerfstudio container, run the following to export the point cloud:
    ```
    ns-export pointcloud --load-config outputs/nerfacto/[session]/config.yml \
    --output-dir . \
    --bounding-box-min -5 -5 -5 \
    --bounding-box-max 5 5 5 \
    --num-points 10000
    ```
    This should generate a `point_cloud.ply` file in your `[data_dir]/[scene]` directory.

## NeRF rendering

1. 




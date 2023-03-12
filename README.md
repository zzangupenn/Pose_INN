# Pose_INN: Visual Pose Regression and Localization with Invertible Neural Networks
## Supplementary Material Readme

Thank you for reviewing my paper. 

In this supplementary material we provide a guidance on reproducing our work. The data preparation part takes about 1 hour. For the Pose_INN training, you can get a good result with 5-6 hours. Due to the file size limit, we can't share the custom data recorded with the real-world robot. This guide will be about how to train on the Cambridge and 7scene dataset.

There are four stages in reproducing our work: 
1. Data processing
2. NeRF training
3. NeRF rendering
4. Pose_INN training

## Data processing

1. You can use our Dockerfile build and run the container:
    ```
    cd Pose_INN
    docker build -t pose_inn .
    docker run -ti --rm --gpus all --ipc=host --name pose_inn \
    -v [data_dir]:/workspace/data \
    -v ./results:/workspace/results pose_inn /bin/bash
    ```
    If you are using a server with multiple users, you need to replace the `1000` in the Dockerfile with your `$UID`. Same goes for the Nerfstudio Dockerfile.

    If you want to visualize the camera sampling result, you may want to run natively.

1. Download the [Cambridge](https://www.repository.cam.ac.uk/handle/1810/251342) or the [7scene](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset. Extract in directory `[data_dir]/[scene]`. e.g. If your downloaded ShopFacade, your `[data_dir]/[scene]` is `[data_dir]/ShopFacade`:
    ```
    cd data
    wget <dataset_url>
    unzip [scene].zip
    ```

2. We use [kapture](https://github.com/naver/kapture) to convert the dataset format to COLMAP format. Run the `kapture_cambridge.sh` or `kapture_7scene.sh` with `[scene]` as argument:
    ```
    cd /workspace
    sh ./kapture_cambridge.sh [data_dir]/[scene]
    ```

3. We will prepare the training data for NeRF: 
    ```
    python3 data_processing_for_nerf.py [scene]
    ```

It will create a directory `[data_dir]/[scene]/images`, a json file `[data_dir]/[scene]/transforms.json`, and a npz file `[data_dir]/[scene]/[scene]_H_matrixes.npz`. Exit the docker container after finish.



## NeRF training

1. We uses [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/v0.1.16) for training the NeRF. Notice the commands provided here may only work with v0.1.16. We attached the version we use. We used their Dockerfile to create a docker container. You can also refer to their wonderful [documentation](https://github.com/nerfstudio-project/nerfstudio/blob/v0.1.16/docs/quickstart/installation.md).

    You need to have cuda version newer than 11.7. Your can use [their docker image](https://hub.docker.com/layers/dromni/nerfstudio/0.1.16/images/sha256-de540fc3e53b62428a4787de78e09feffc84cfbadcca6b4afe4df40a78d3fd92?context=explore):
    
    ```
    docker pull dromni/nerfstudio:0.1.16
    docker run --gpus all --network host --name pose_inn_nerfstudio \
        -v [data_dir]/:/workspace/ \
        -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it dromni/nerfstudio:0.1.16
    ```

    Or build and run a local docker image:
    ```
    cd nerfstudio
    docker build --tag nerfstudio -f Dockerfile .
    docker run --gpus all --network host --name pose_inn_nerfstudio \
        -v [data_dir]/[scene]/:/workspace/ \
        -v ~/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it nerfstudio
    ```

2. Inside the container, start the NeRF training:
    ```
    ns-train nerfacto --data . --max-num-iterations 60000 nerfstudio-data --train-split-percentage 1  
    ```
    You should be able to see the training progress with their viewer. 
    
    One thing we noticed is that the training sometimes is not stable. If it crashes or the viewer only renders black, you need to retrain the model.  

3. After the training, you should see a `outputs/nerfacto/` folder in your `[data_dir]/[scene]` directory with your trained sessions. We will refer that as the `[session]` folder. e.g. A `[session]` folder may look like `2023-03-11_062711`.

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

1. We go back to the Pose_INN container and sample camera poses:
    ```
    python3 camera_sampling.py [scene] [session]
    ```

2. In the nerfstudio container, run the following to render the images:
    ```
    ns-render --load-config outputs/nerfacto/[session]/config.yml \
    --traj filename \
    --camera-path-filename ./camera_path.json \
    --output-path ./render/ \
    --output-format images
    ```
    This puts the rendered image in `[data_dir]/[scene]/render/`.

2. Modify the first two lines in `data_gathering.py` and run it.
    This will generate a `50k_train_w_render.npz` file in your `[data_dir]/[scene]`.

## Pose_INN training.

1. Please use the Dockerfile build and run the container:
    ```
    cd Pose_INN
    docker build -t pose_inn .
    docker run -ti --gpus all --ipc=host --name pose_inn \
    -v [data_dir]:/workspace/data \
    -v ./results:/workspace/result pose_inn /bin/bash
    ```




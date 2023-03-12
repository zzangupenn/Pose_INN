docker build -t pose_inn .

docker run --gpus '"device=0"' --network host --name pose_inn_nerfstudio_0 \
        -v ~/data/pose_inn/fire/:/workspace/ \
        -v ~/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it nerfstudio


docker run -ti --rm --gpus all --ipc=host --name pose_inn_0 \
    -v ~/data/pose_inn:/workspace/data \
    -v ./results:/workspace/results pose_inn /bin/bash

ns-export pointcloud --load-config outputs/nerfacto/2023-03-12_063624/config.yml \
    --output-dir . \
    --bounding-box-min -5 -5 -5 \
    --bounding-box-max 5 5 5 \
    --num-points 10000


python3 camera_sampling.py fire 2023-03-12_063624


    ns-render --load-config outputs/nerfacto/2023-03-11_233258/config.yml \
    --traj filename \
    --camera-path-filename ./camera_path.json \
    --output-path ./render/ \
    --output-format images
    ```

python3 data_gathering.py KingsCollege

python3 Pose_INN.py pose_inn_kings_r2.py


fire 2023-03-12_063624
KingsCollege 2023-03-11_233258
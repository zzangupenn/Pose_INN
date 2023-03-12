docker run --gpus '"device=1"' --network host --name pose_inn_nerfstudio \
        -v ~/data/pose_inn/KingsCollege/:/workspace/ \
        -v ~/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it nerfstudio


docker run -ti --rm --gpus all --ipc=host --name pose_inn \
    -v ~/data/pose_inn:/workspace/data \
    -v ./results:/workspace/results pose_inn /bin/bash


python3 camera_sampling.py KingsCollege 2023-03-11_233258


    ns-render --load-config outputs/nerfacto/2023-03-11_233258/config.yml \
    --traj filename \
    --camera-path-filename ./camera_path.json \
    --output-path ./render/ \
    --output-format images
    ```

python3 data_gathering.py KingsCollege
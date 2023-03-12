docker build -t pose_inn .


docker run -ti --rm --gpus all --ipc=host --name pose_inn_0 \
    -v ~/data/pose_inn:/workspace/data \
    -v ~/results2/pose_inn/pose_inn_0:/workspace/results pose_inn /bin/bash
    

sh ./kapture_cambridge.sh data/[scene]

python3 data_processing_for_nerf.py OldHospital


docker run --gpus '"device=2"' --network host --name pose_inn_nerfstudio_2 \
        -v ~/data/pose_inn/OldHospital/:/workspace/ \
        -v ~/.cache/:/home/user/.cache/ \
        -p 7007:7007 --shm-size=12gb --rm -it nerfstudio

ns-train nerfacto --data . --max-num-iterations 60000 nerfstudio-data --train-split-percentage 1  

ns-export pointcloud --load-config outputs/nerfacto/2023-03-12_063624/config.yml \
    --output-dir . \
    --bounding-box-min -5 -5 -5 \
    --bounding-box-max 5 5 5 \
    --num-points 10000




python3 camera_sampling.py fire 2023-03-12_063624


ns-render --load-config outputs/nerfacto/2023-03-12_063624/config.yml \
    --traj filename \
    --camera-path-filename ./camera_path.json \
    --output-path ./render/ \
    --output-format images

python3 data_gathering.py KingsCollege

python3 Pose_INN.py KingsCollege pose_inn_kings cuda:0


fire 2023-03-12_063624
KingsCollege 2023-03-11_233258
OldHospital
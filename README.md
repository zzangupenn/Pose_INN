# Pose_INN: Visual Pose Regression and Localization with Invertible Neural Networks
## Supplementary Material Readme

Thank you for reviewing my paper. 

In this supplementary material we provide a guidance on reproducing our work. Our data sampling and training process is fast. The data preparation part takes about 1 hour. For the Pose_INN training, you can get a good result for 5-6 hours. Due to the file size limit, we can't share the custom data with you. This guide will about how to train on the Cambridge and 7scene dataset.

There are three stages in reproducing our work: 
1. Data processing
2. NeRF training and rendering
3. Pose_INN training

## Data processing

1. Download the [Cambridge](https://www.repository.cam.ac.uk/handle/1810/251342) or the [7scene](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset. Extract in directory [data_dir]/[scene]. e.g. If your downloaded ShopFacade, your [data_dir]/[scene] is [data_dir]/ShopFacade. If your are testing on 7scene, extract all the seq zip files in [data_dir]/[scene].

2. We use [kapture](https://github.com/naver/kapture) to convert the dataset format to COLMAP format. A version we used is attached, but it can be install with:
    ```
    pip3 install kapture
    ```
    Copy the `kapture_cambridge.sh` or `kapture_7scene.sh` into [data_dir]/[scene] and run it.

3. Modify the first two lines in `data_processing_cambridge_for_nerf.py` or `data_processing_7scene_for_nerf.py` and run it. It create a directory `[data_dir]/[scene]/images`, a json file 

4. 

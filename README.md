# Pose_INN: Visual Pose Regression and Localization with Invertible Neural Networks
## Supplementary Material Readme

Thank you for reviewing my paper. 

In this supplementary material we provide a guidance on reproducing our work. Our data sampling and training process is fast. The data preparation part takes about 1 hour. For the Pose_INN training, you can get a good result for 5-6 hours. Due to the file size limit, we can't share the custom data with you. This guide will about how to train on the Cambridge and 7scene dataset.

There are three stages in reproducing our work: 
1. Data processing
2. NeRF training and rendering
3. Pose_INN training

## Data processing

1. download the [Cambridge](https://www.repository.cam.ac.uk/handle/1810/251342) or the [7scene](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset. Extract in directory [data_dir].

2. We use [kapture](https://github.com/naver/kapture) to convert the dataset format to COLMAP format. A attached a version we used.
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import trange
import os, sys, cv2
import utils

data_dir = 'data'
scene = sys.argv[1]
data_dir = data_dir + '/' + scene

print('Load poses')
data_dir = data_dir + '/' + scene
data_dir_render = data_dir + '/render'
data_npz = np.load(data_dir + '/' + scene + '_H_matrixes.npz')
train_H_matrixes = data_npz['train_H_matrixes']
test_H_matrixes = data_npz['test_H_matrixes']
print(train_H_matrixes.shape)
print(test_H_matrixes.shape)



print('Gather data')
train_imgs = []
train_poses = []
for ind in trange(train_H_matrixes.shape[0]):
    ## SfM poses in training set is wrong
    if scene == 'ShopFacade' and ind in [215, 216]:
        continue
    elif scene == 'StMarysChurch' and ind in [121]:
        continue
    img = cv2.resize(cv2.imread(data_dir + "/images/{:05d}.png".format(ind)), (160, 90))
    train_imgs.append(np.asarray(img))
    train_poses.append(np.concatenate([train_H_matrixes[ind, :3, 3], R.from_matrix(train_H_matrixes[ind, :3, :3]).as_euler('zxy', degrees=False)]))

test_imgs = []
test_poses = []
for ind in trange(test_H_matrixes.shape[0]):
    img = cv2.resize(cv2.imread(data_dir + "/images/{:05d}.png".format(ind + train_H_matrixes.shape[0])), (160, 90))
    test_imgs.append(np.asarray(img))
    test_poses.append(np.concatenate([test_H_matrixes[ind, :3, 3], R.from_matrix(test_H_matrixes[ind, :3, :3]).as_euler('zxy', degrees=False)]))

camera_json = utils.ConfigJSON()
camera_json.load_file(data_dir + '/camera_path_original.json')
camera_path = camera_json.d["camera_path"]

for ind in trange(len(camera_path)):
    H_matrix = np.array(camera_path[ind]['camera_to_world']).reshape(4, 4)
    train_poses.append(np.concatenate([H_matrix[:3, 3], 
                                      R.from_matrix(H_matrix[:3, :3]).as_euler('zxy', degrees=False)]))
    img = cv2.imread(data_dir_render + "/{:05d}.png".format(ind))
    train_imgs.append(np.asarray(img))
    
train_imgs = np.array(train_imgs)
train_poses = np.array(train_poses)
test_imgs = np.array(test_imgs)
test_poses = np.array(test_poses)
print(train_imgs.shape)
print(train_poses.shape)
print(test_imgs.shape)
print(test_poses.shape)

## mean filter for failed rendering
bad_img_inds = []
for ind in trange(train_H_matrixes.shape[0], train_imgs.shape[0]):
    if np.mean(train_imgs[ind]) < 50 or np.mean(train_imgs[ind]) > 220:
        bad_img_inds.append(ind)
train_imgs = np.delete(train_imgs, bad_img_inds, axis=0)
train_poses = np.delete(train_poses, bad_img_inds, axis=0)
print(train_imgs.shape)
print(train_poses.shape)

np.savez(data_dir + '/50k_train_w_render.npz', 
                    train_imgs = np.array(train_imgs),
                    train_poses = np.array(train_poses),
                    test_imgs = np.array(test_imgs),
                    test_poses = np.array(test_poses))
        




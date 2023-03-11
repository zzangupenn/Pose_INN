import numpy as np
import os, sys, cv2
import utils, rotation_utils
from tqdm import trange

data_dir = 'data'
scene = sys.argv[1]
data_dir = data_dir + '/' + scene
data_dir_train = data_dir + '/kapture/mapping/sensors'
data_dir_test = data_dir + '/kapture/query/sensors'
if not os.path.exists(data_dir + '/images'):
    os.mkdir(data_dir + '/images')
    


print('Save the images from original folders.')
lines_train = utils.readTXT(data_dir_train + '/records_camera.txt')
for ind in trange(2, len(lines_train)):
    line = lines_train[ind].strip().split(', ')
    img = cv2.imread(data_dir_train + '/records_data/' + line[2])
    cv2.imwrite(data_dir + '/' + "images/{:05d}.png".format(ind-2), img)

lines = utils.readTXT(data_dir_test + '/records_camera.txt')
for ind in trange(2, len(lines)):
    line = lines[ind].strip().split(', ')
    img = cv2.imread(data_dir_test + '/records_data/' + line[2])
    cv2.imwrite(data_dir + '/' + "images/{:05d}.png".format(ind-2 + len(lines_train)-2), img)
    
    
    
    
print('Create json files for nerf.')
lines = utils.readTXT(data_dir_train + '/trajectories.txt')
transforms_json = utils.ConfigJSON()
if scene in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
    transforms_json.d["fl_x"] = 1670
    transforms_json.d["fl_y"] = 1670
    transforms_json.d["k1"] = 0
    transforms_json.d["k2"] = 0
    transforms_json.d["p1"] = 0
    transforms_json.d["p2"] = 0
    transforms_json.d["cx"] = 960
    transforms_json.d["cy"] = 540
    transforms_json.d["w"] = 1920
    transforms_json.d["h"] = 1080
    transforms_json.d["aabb_scale"] = 2
elif scene in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    transforms_json.d["fl_x"] = 525
    transforms_json.d["fl_y"] = 525
    transforms_json.d["k1"] = 0
    transforms_json.d["k2"] = 0
    transforms_json.d["p1"] = 0
    transforms_json.d["p2"] = 0
    transforms_json.d["cx"] = 320
    transforms_json.d["cy"] = 240
    transforms_json.d["w"] = 640
    transforms_json.d["h"] = 480
    transforms_json.d["aabb_scale"] = 16
transforms_json.d["frames"] = []

for ind in trange(2, len(lines)):
    if scene == 'ShopFacade' and ind in [215+2, 216+2]: ## SfM pose in training set is wrong
        continue
    frame_dict = {}
    frame_dict["file_path"] = "./images/{:05d}.png".format(ind-2)
    pose = np.array(lines[ind].strip().split(', ')[2:]).astype(np.float64)
    rotation = rotation_utils.qvec2rotmat([pose[0], pose[1], pose[2], pose[3]])
    translation = np.array([pose[4], pose[5], pose[6]]).reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system to ours
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    frame_dict["transform_matrix"] = c2w.tolist()
    transforms_json.d["frames"].append(frame_dict)
transforms_json.save_file(data_dir + '/transforms.json')




print('Save the poses')
lines = utils.readTXT(data_dir_train + '/trajectories.txt')
train_H_matrixes = []
for ind in trange(2, len(lines)):
    pose = np.array(lines[ind].strip().split(', ')[2:]).astype(np.float64)
    rotation = rotation_utils.qvec2rotmat([pose[0], pose[1], pose[2], pose[3]])
    translation = np.array([pose[4], pose[5], pose[6]]).reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system to nerf
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    train_H_matrixes.append(c2w)
train_H_matrixes = np.array(train_H_matrixes)
print('train_H_matrixes', train_H_matrixes.shape)

lines = utils.readTXT(data_dir_test + '/trajectories.txt')
test_H_matrixes = []
for ind in trange(2, len(lines)):
    pose = np.array(lines[ind].strip().split(', ')[2:]).astype(np.float64)
    rotation = rotation_utils.qvec2rotmat([pose[0], pose[1], pose[2], pose[3]])
    translation = np.array([pose[4], pose[5], pose[6]]).reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system to nerf
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    test_H_matrixes.append(c2w)
test_H_matrixes = np.array(test_H_matrixes)
print('test_H_matrixes', test_H_matrixes.shape)

np.savez(data_dir + '/' + scene + '_H_matrixes.npz', train_H_matrixes=train_H_matrixes, test_H_matrixes=test_H_matrixes)

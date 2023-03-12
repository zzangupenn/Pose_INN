import open3d as o3d
import numpy as np
import os, sys, cv2, copy
import utils, rotation_utils
from tqdm import trange
from scipy.spatial.transform import Rotation as R

data_dir = 'data'
scene = sys.argv[1]
data_dir = data_dir + '/' + scene
session = sys.argv[2] 


if scene in ['ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch']:
    fov = 36.039642368
    resolution = [160, 90]
elif scene in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    fov = 62.7269
    resolution = [160, 120]

if scene == 'ShopFacade':
    R_to_align_axis = R.from_euler('z', 0, degrees=True).as_matrix()
    delta_training = 5
    N_in_view = [7500, 20000]
    delta_in_view = [1.2, 8]
elif scene == 'KingsCollege':
    R_to_align_axis = R.from_euler('x', -0.5, degrees=True).as_matrix()
    delta_training = 5
    N_in_view = [1500, 18000]
    delta_in_view = [1, 17]
elif scene == 'OldHospital':
    R_to_align_axis = R.from_euler('yx', [-6, -9], degrees=True).as_matrix()
    delta_training = 7
    N_in_view = [3000, 23000]
    delta_in_view = [0.5, 23]
elif scene == 'StMarysChurch':
    R_to_align_axis = R.from_euler('z', -2, degrees=True).as_matrix()
    delta_training = 5
    N_in_view = [2000, 13000]
    delta_in_view = [2.5, 15]

def angle_w_z(point):
    return np.arcsin(np.sqrt(point[:, 0] ** 2 + point[:, 1] ** 2) / np.linalg.norm(point, axis=1))

def camera_sample(sampled_num, R_to_align_axis, train_H_matrixes, test_H_matrixes,
                  delta_training, N_in_view, delta_in_view, pcd):
    
    sampled_Hs = []
    cnt = 0
    printed_cnt = -1

    ## rotation the poses so that it's easier to sample
    aligned_train_pos = []
    aligned_test_pos = []
    for ind in range(len(train_H_matrixes)):
        aligned_train_pos.append(R_to_align_axis @ train_H_matrixes[ind, :3, 3])
    for ind in range(len(test_H_matrixes)):
        aligned_test_pos.append(R_to_align_axis @ test_H_matrixes[ind, :3, 3])
    aligned_train_pos = np.array(aligned_train_pos)
    aligned_test_pos = np.array(aligned_test_pos)

    xyz_range = utils.find_larger_range(utils.find_range(aligned_train_pos), 
                                        utils.find_range(aligned_test_pos))
    
    while cnt < sampled_num:
        if cnt % 500 == 0 and printed_cnt < cnt:
            print(cnt)
            printed_cnt = cnt
            
        sampled_xyz = np.array([np.random.uniform(xyz_range[0, 0], xyz_range[1, 0], 1),
                                np.random.uniform(xyz_range[0, 1], xyz_range[1, 1], 1),
                                np.random.uniform(xyz_range[0, 2], xyz_range[1, 2], 1)])
    
        distances_trainset = np.linalg.norm(aligned_train_pos - sampled_xyz.T, axis=1)
        if np.min(distances_trainset) > delta_training: 
            continue
        
        sampled_H = np.eye(4)
        sampled_H[:3, :3] = rotation_utils.uniform_random_rotation(0.01) @ train_H_matrixes[np.random.choice(np.where(distances_trainset < 10)[0], 1), :3, :3]
        sampled_H[:3, 3] = (np.linalg.inv(R_to_align_axis) @ sampled_xyz).T

        pcd2 = copy.deepcopy(pcd)
        pcd2.transform(np.linalg.inv(sampled_H))
        max_bound = pcd2.get_max_bound()
        max_bound[2] = 0
        try:
            pcd2 = pcd2.crop(o3d.geometry.AxisAlignedBoundingBox(pcd2.get_min_bound(), max_bound))
        except:
            continue
        points = np.asarray(pcd2.points)
        fov = 36 / 180 * np.pi / 2 
        pcd2 = pcd2.select_by_index(np.where(angle_w_z(points) < fov)[0])

        pcd2_xyz = np.asarray(pcd2.points)
        distances = np.linalg.norm(pcd2_xyz, axis=1)
        
        if np.asarray(pcd2.points).shape[0] > N_in_view[0] and np.asarray(pcd2.points).shape[0] < N_in_view[1]:
            if np.min(distances) > delta_in_view[0] and np.min(distances) < delta_in_view[1]:
                sampled_Hs.append(sampled_H)
                cnt += 1
        
    return np.array(sampled_Hs)



print('Load poses')
data_npz = np.load(data_dir + '/' + scene + '_H_matrixes.npz')
train_H_matrixes = data_npz['train_H_matrixes']
test_H_matrixes = data_npz['test_H_matrixes']
## SfM poses in training set is wrong
if scene == 'ShopFacade':
    train_H_matrixes = np.delete(train_H_matrixes, [215, 216], axis=0) # shop
elif scene == 'StMarysChurch':
    train_H_matrixes = np.delete(train_H_matrixes, [121], axis=0) # church
print(train_H_matrixes.shape)
print(test_H_matrixes.shape)



print('Load point cloud')
scaling_json = utils.ConfigJSON()
scaling_json.load_file(data_dir + '/outputs/nerfacto/' + session + '/dataparser_transforms.json')
transform = np.concatenate([np.array(scaling_json.d["transform"]), [[0, 0, 0, 1]]])
scale = scaling_json.d["scale"]

pcd = o3d.io.read_point_cloud(data_dir + '/point_cloud.ply')
pcd.scale(1/scale, center=[0, 0, 0])
pcd.transform(np.linalg.inv(transform))
print(pcd)



print('Show the statistics of train poses. Omitted')
# point_counts = []
# distance_counts = []
# for ind in trange(train_H_matrixes.shape[0]):
#     pcd2 = copy.deepcopy(pcd)
#     pcd2.transform(np.linalg.inv(train_H_matrixes[ind]))
#     max_bound = pcd2.get_max_bound()
#     max_bound[2] = 0
#     try:
#         pcd2 = pcd2.crop(o3d.geometry.AxisAlignedBoundingBox(pcd2.get_min_bound(), max_bound))
#     except:
#         continue
#     points = np.asarray(pcd2.points)
#     angles = angle_w_z(points)
#     fov = 36 / 180 * np.pi / 2
#     pcd2 = pcd2.select_by_index(np.where(angle_w_z(points) < fov)[0])
#     pcd2_xyz = np.asarray(pcd2.points)
#     if pcd2_xyz.shape[0] > 0:
#         distances = np.linalg.norm(pcd2_xyz, axis=1)
#         point_counts.append(pcd2_xyz.shape[0])
#         distance_counts.append(np.min(distances))
#     # print(pcd2_xyz.shape[0], np.min(distances))
# import matplotlib.pyplot as plt
# plt.hist(point_counts, bins='auto')
# plt.show()
# plt.hist(distance_counts, bins='auto')
# plt.show()
# print(np.median(point_counts))
# print(np.median(distance_counts))



print('First sample 1000 poses for visualization. Omitted.')
# sampled_num = 1000
# sampled_Hs = camera_sample(sampled_num, R_to_align_axis, train_H_matrixes, test_H_matrixes,
#                         delta_training, N_in_view, delta_in_view, pcd)

# o3d_visualizer = utils.open3dUtils()
# o3d_visualizer.show_axis = True
# size = 0.5
# for ind in range(sampled_Hs.shape[0]):
#     o3d_visualizer.add_object(o3d_visualizer.create_camera_poses(sampled_Hs[ind], size=size, color=[1.0, 0.5, 0]))
# o3d_visualizer.add_object(pcd.select_by_index(np.where(np.mean(np.asarray(pcd.colors), axis=1) < 0.9)[0]))
# # o3d_visualizer.add_object(pcd)

# for ind in np.random.choice(np.arange(train_H_matrixes.shape[0]), np.min([500, train_H_matrixes.shape[0]]), replace=False):
#     H_matrix = train_H_matrixes[ind].copy()
#     o3d_visualizer.add_object(o3d_visualizer.create_camera_poses(H_matrix, size=size, color=[1, 0, 0]))
    
# for ind in np.random.choice(np.arange(test_H_matrixes.shape[0]), np.min([500, test_H_matrixes.shape[0]]), replace=False):
#     H_matrix = test_H_matrixes[ind].copy()
#     o3d_visualizer.add_object(o3d_visualizer.create_camera_poses(H_matrix, size=size, color=[0, 0, 1]))
# o3d_visualizer.show()



print('Sample 50000 poses for visualization. This will take 10 minutes')
sampled_num = 500
sampled_Hs = camera_sample(sampled_num, R_to_align_axis, train_H_matrixes, test_H_matrixes,
                        delta_training, N_in_view, delta_in_view, pcd)



print('Output as JSON file')
camera_json = utils.ConfigJSON()
camera_json.d["camera_type"] = "perspective"
camera_json.d["render_height"] = resolution[1]
camera_json.d["render_width"] = resolution[0]
camera_json.d["fps"] = 24
camera_json.d["seconds"] = 4
camera_json.d["smoothness_value"] = 0.5
camera_json.d["is_cycle"] = False
camera_json.d["crop"] = None

camera_json_original = utils.ConfigJSON()
camera_path = []
camera_path_original = []
for ind in range(sampled_Hs.shape[0]):
    camera_path_json = {}
    H_matrix = sampled_Hs[ind].copy()
    camera_path_json["camera_to_world"] = list(H_matrix.flatten())
    camera_path_original.append(camera_path_json)
    
    camera_path_json = {}
    H_matrix = sampled_Hs[ind].copy()
    H_matrix = transform @ H_matrix
    H_matrix[:3, 3] *= scale
    camera_path_json["camera_to_world"] = list(H_matrix.flatten())
    camera_path_json["fov"] = fov
    camera_path.append(camera_path_json)
    
camera_json.d["camera_path"] = camera_path
camera_json.save_file(data_dir + '/camera_path.json')    

camera_json_original.d["camera_path"] = camera_path_original
camera_json_original.save_file(data_dir + '/camera_path_original.json')  
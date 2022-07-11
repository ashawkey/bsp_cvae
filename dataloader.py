import torch.utils.data
from torch.utils.data import DataLoader, Dataset

import os
import tqdm
import json
import pickle
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

import trimesh

from utils import AttributeDict

import glob

###################################
## some constants
###################################

MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])

ShapeNetIDMap = {
    '4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', '4460130': 'tower', '3001627': 'chair', 
    '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', 
    '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', '3710193': 'mailbox', 
    '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', 
    '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', 
    '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', 
    '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', 
    '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}                   

CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
labels2id = {v: k for k, v in enumerate(CAD_labels)}
CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])} # selected 8 categories from SHAPENETCLASSES
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

###################################
## utils
###################################

def read_json(file):
    with open(file, 'r') as f:
        output = json.load(f)
    return output

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

class SubsamplePoints(object):

    def __init__(self, N, mode):
        self.N = N
        self.mode = mode

    def __call__(self, data):
        points = data['points']
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            if self.mode == 'test':
                idx = np.arange(0, self.N)
            else:
                idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'points': points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            if self.mode == 'test':
                idx0 = np.arange(0, Nt_out)
                idx1 = np.arange(0, Nt_in)
            else:
                idx0 = np.random.randint(points0.shape[0], size=Nt_out)
                idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                'points': points,
                'occ': occ,
                'volume': volume,
            })
        return data_out    

###################################
## ShapeNet dataset
###################################

class ShapeNetDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.augment = mode == 'train'
        self.data_root = cfg['data_root']
        #self.num_points = cfg['num_point']
        self.points_unpackbits = cfg['points_unpackbits']
        self.transform = SubsamplePoints(cfg['points_subsample'], mode)
        self.voxel_resolution = np.array(cfg['voxel_resolution'])
        self.shapenet_path = os.path.join(self.data_root, 'datasets/ShapeNetv2_data')
        self.transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

        # glob all data path
        self.files = glob.glob(os.path.join(self.shapenet_path, 'point', '*', '*.npz'))

        if mode != 'train' and cfg['fast_eval']:
            #np.random.shuffle(self.files)
            #self.files = self.files[3000:3100]
            self.files = self.files[:10]
        

        # pre-loading
        if self.mode == 'train':
            self.points_dict = []
            print('[INFO] pre-loading datasets...')
            for i in tqdm.trange(len(self.files)):
                d = np.load(self.files[i])
                self.points_dict.append(d)

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        path = self.files[idx]
        if self.mode == 'train':
            points_dict = self.points_dict[idx]
        else:
            points_dict = np.load(path)

        ### name
        name = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))

        ### load point cloud input
        #point_cloud_dict = np.load(path.replace('point', 'pointcloud'))
        # for input, we need to use the pointcloud, and sparse quantize it.
        #point_cloud = point_cloud_dict['points'] # [N, 3], N = 100000
        #point_normals = point_cloud_dict['normals'] # [N, 3]

        ### load training point-occ pairs
        points = points_dict['points'] # range (-0.55, 0.55)
        # Break symmetry if given in float16:
        if points.dtype == np.float16 and self.mode == 'train':
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        # swap axes 
        points = points.dot(self.transform_m.T)

        occupancies = points_dict['occupancies']
        if self.points_unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # random rotate 360
        #orientation = np.random.randn(1)[0] * np.pi
        #axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        #points = points.dot(axis_rectified)

        # rescale
        grid_length = 2 / self.voxel_resolution 
        half_grid_length = 1 / self.voxel_resolution 
        min_xyz = points[occupancies == 1].min(0)
        max_xyz = points[occupancies == 1].max(0)

        ### load voxels
        points = (2 * (points - min_xyz) / (max_xyz - min_xyz + 1e-5) - 1) * (1 - half_grid_length)

        # mask out outlier points caused by rotation
        mask = np.logical_and(points <= 1, points >= -1).sum(axis=1) == 3
        points = points[mask]
        occupancies = occupancies[mask]

        occ_points = points[occupancies == 1]
        
        # voxelize it
        #print(points.min(axis=0), points.max(axis=0))
        #print(occ_points.min(axis=0), occ_points.max(axis=0))
        point_coords = np.floor((occ_points + 1) / grid_length).astype(np.int32) # [n, 3], in [0, vox_res)
        
        # encode the voxel
        voxels = np.zeros(list(self.voxel_resolution) + [4], dtype=np.float32)
        features = np.concatenate([np.ones((occ_points.shape[0], 1)), occ_points], axis=-1) # [N, 4]
        voxels[tuple(point_coords.T)] = features
        voxels = voxels.transpose(3,0,1,2)

        # subsample: each object 100000 points, half on surface, half uniform --> subsample to 2048 + 2048 points for training.
        data = {'points': points, 'occ': occupancies}
        if self.transform is not None:
            data = self.transform(data)

        data['voxels'] = voxels
        data['labels'] = np.array(labels2id[ShapeNetIDMap[label[1:]]])
        data['name'] = name

        return data

###################################
## ScanNet dataset
###################################

class ScanNetDataset(Dataset):
    def __init__(self, cfg, mode, split_name=None):
        super(ScanNetDataset, self).__init__()
        
        self.cfg = cfg
        self.mode = mode
        self.split_name = split_name if split_name is not None else mode
        self.augment = mode == 'train'
        self.data_root = cfg['data_root']
        
        self.shapenet_path = os.path.join(self.data_root, 'datasets/ShapeNetv2_data')
        self.pointgroup_path = os.path.join(self.data_root, 'datasets/pointgroup_output')

        self.split = read_json(os.path.join(self.data_root, 'datasets/splits/fullscan', 'scannetv2_' + self.split_name + '.json'))

        # tmp
        if self.cfg.fast_eval and self.mode == 'val':
            self.split = self.split[:100]

        self.num_points = cfg['num_point']
        self.use_color = cfg['use_color']
        self.use_height = not cfg['no_height']
        self.points_unpackbits = cfg['points_unpackbits']
        self.n_points_object = cfg['points_subsample'] # [1024, 1024] for in_on / out point
        self.points_transform = SubsamplePoints(cfg['points_subsample'], mode)
        self.voxel_resolution = np.array(cfg['voxel_resolution'])
        self.transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        data_path = self.split[idx] # json format

        # datasets/scannet/processed_data/scene0100_01/full_scan.npz
        scan_name = data_path['scan'].split('/')[-2]

        # load bbox
        with open(os.path.join(self.data_root, data_path['bbox']), 'rb') as file:
            box_info = pickle.load(file)
        boxes3D = []
        classes = [] # class of bbox
        shapenet_catids = []
        shapenet_ids = []
        object_instance_ids = []
        for item in box_info:
            object_instance_ids.append(item['instance_id'])
            boxes3D.append(item['box3D'])
            classes.append(item['cls_id'])
            shapenet_catids.append(item['shapenet_catid'])
            shapenet_ids.append(item['shapenet_id'])
        boxes3D = np.array(boxes3D)

        # load fullscan
        scan_data = np.load(os.path.join(self.data_root, data_path['scan']))
        point_cloud = scan_data['mesh_vertices']
        # point_instance_labels = scan_data['instance_labels']
        # point_semantic_labels = np.load(data_path['scan'].replace('full_scan.npz', 'semantics.npz'))['semantic_labels']

        # input point cloud features
        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]  # do not use color for now
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        # augmentation point cloud and bbox
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                boxes3D[:, 0] = -1 * boxes3D[:, 0]
                boxes3D[:, 6] = np.sign(boxes3D[:, 6]) * np.pi - boxes3D[:, 6]
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                boxes3D[:, 1] = -1 * boxes3D[:, 1]
                boxes3D[:, 6] = -1 * boxes3D[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4  # -45 ~ +45 degree
            rot_mat = rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            boxes3D[:, 0:3] = np.dot(boxes3D[:, 0:3], np.transpose(rot_mat))
            boxes3D[:, 6] += rot_angle

            # Normalize angles to [-pi, pi]
            boxes3D[:, 6] = np.mod(boxes3D[:, 6] + np.pi, 2 * np.pi) - np.pi

        bboxes_label = np.array([ShapeNet2CAD[x] for x in classes]) # bbox class to [0, 7]

        # bbox related variables
        # target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        # target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        # target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        # object_instance_labels = np.zeros((MAX_NUM_OBJ, ))

        # target_bboxes_mask[0:boxes3D.shape[0]] = 1 # to distinguish paddings
        # target_bboxes[0:boxes3D.shape[0], :] = boxes3D[:,0:6]
        # target_bboxes_semcls[0:boxes3D.shape[0]] = bboxes_label 
        # object_instance_labels[0:boxes3D.shape[0]] = object_instance_ids # == target_bboxes_inscls

        point_cloud, choices = self.random_sampling(point_cloud, self.num_points, return_choices=True)
        # # point_instance_labels = # point_instance_labels[choices]
        # point_semantic_labels = point_semantic_labels[choices] 

        # prepare output dict
        ret_dict = {}

        # pano seg
        # ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        # # ret_dict['semantic_labels'] = point_semantic_labels.astype(np.int64)
        # ret_dict['instance_labels'] = # point_instance_labels.astype(np.int64)

        # bbox 
        # ret_dict['boxes'] = target_bboxes.astype(np.float32) # [M, 6]
        # ret_dict['boxes_mask'] = target_bboxes_mask.astype(np.float32) # [M]
        # ret_dict['box_semantic_labels'] = target_bboxes_semcls.astype(np.int64) # [M]
        
        # idx
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['scan_name'] = scan_name
    
        ### load shapenet GT for completion   
        if self.mode == 'train':
            MAX_NUM_OBJ_pg = 16
        else:
            MAX_NUM_OBJ_pg = 64

        # randomize order
        if self.mode == 'train' and boxes3D.shape[0] > MAX_NUM_OBJ_pg:
            candidates = np.random.choice(boxes3D.shape[0], MAX_NUM_OBJ_pg, replace=False)
        else:
            candidates = np.arange(boxes3D.shape[0])
        
        # sample training points for OccNet (this is from shapenet GT)
        object_mask = np.zeros((MAX_NUM_OBJ_pg), dtype=np.uint8) # [M]
        object_points = np.zeros((MAX_NUM_OBJ_pg, np.sum(self.n_points_object), 3)) # [M, I+O, 3], In = Out = 1024
        object_points_occ = np.zeros((MAX_NUM_OBJ_pg, np.sum(self.n_points_object))) # [M, I+O], indicates In (1) or Out (0) for OccNet
        object_bbox = np.zeros((MAX_NUM_OBJ_pg, 7), dtype=np.float32) # [M, 7] oriented bbox
        object_semantic_labels = np.zeros((MAX_NUM_OBJ_pg), dtype=np.int64) # [M]
        object_voxels = np.zeros((MAX_NUM_OBJ_pg, 4, *self.voxel_resolution), dtype=np.float32) # [M, 4/5, H, W, D]

        points_data = self.get_shapenet_points(shapenet_catids, shapenet_ids, transform=self.points_transform) # {points: [M, n, 3], occ: [M, n]}, subsampled N = I+O

        # fit object points to bboxes
        #for i in range(boxes3D.shape[0]):
        cnt = 0
        for i in candidates:

            bbox = boxes3D[i] # [7]

            ### load training point-occ pairs
            points = points_data['points'][i] # range (-0.55, 0.55)
            # Break symmetry if given in float16:
            if points.dtype == np.float16 and self.mode == 'train':
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            # swap axes 
            points = points.dot(self.transform_m.T)

            occupancies = points_data['occ'][i]
            if self.points_unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)

            #orientation = bbox[6] # use the real orientation!!! the only change.
            #axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
            #points = points.dot(axis_rectified)

            # rescale
            grid_length = 2 / self.voxel_resolution 
            half_grid_length = 1 / self.voxel_resolution 
            min_xyz = points[occupancies == 1].min(0)
            max_xyz = points[occupancies == 1].max(0)

            ### load voxels
            points = (2 * (points - min_xyz) / (max_xyz - min_xyz + 1e-5) - 1) * (1 - half_grid_length)

            # mask out outlier points caused by rotation
            mask = np.logical_and(points <= 1, points >= -1).sum(axis=1) == 3
            points = points[mask]
            occupancies = occupancies[mask]

            occ_points = points[occupancies == 1]
            
            # voxelize it
            #print(points.min(axis=0), points.max(axis=0))
            #print(occ_points.min(axis=0), occ_points.max(axis=0))
            point_coords = np.floor((occ_points + 1) / grid_length).astype(np.int32) # [n, 3], in [0, vox_res)
            
            # encode the voxel
            voxels = np.zeros(list(self.voxel_resolution) + [4], dtype=np.float32)
            features = np.concatenate([np.ones((occ_points.shape[0], 1)), occ_points], axis=-1) # [N, 4]
            voxels[tuple(point_coords.T)] = features
            

            # subsample: each object 100000 points, half on surface, half uniform --> subsample to 2048 + 2048 points for training.
            data = {'points': points, 'occ': occupancies}
            if self.points_transform is not None:
                data = self.points_transform(data)
            

            object_mask[cnt] = 1
            object_points[cnt] = data['points']
            object_points_occ[cnt] = data['occ']
            object_semantic_labels[cnt] = bboxes_label[cnt]
            object_bbox[cnt] = bbox # np.concatenate([occ_points.min(0), occ_points.max(0)]) # [6]
            object_voxels[cnt] = voxels.transpose(3,0,1,2)
            cnt += 1

        ret_dict['cnt'] = cnt # [1,]
        ret_dict['points'] = object_points.astype(np.float32) # [M, n, 3]
        ret_dict['points_occ'] = object_points_occ.astype(np.float32) # [M, n]
        ret_dict['bbox'] = object_bbox.astype(np.float32) # [M, 7]
        ret_dict['mask'] = object_mask.astype(np.int64) # [M,]
        ret_dict['voxels'] = object_voxels.astype(np.float32) # [M, H, W, D]
        ret_dict['labels'] = object_semantic_labels.astype(np.int64) # [M,]
        ret_dict['shapenet_catids'] = shapenet_catids
        ret_dict['shapenet_ids'] = shapenet_ids

        return ret_dict

    def get_shapenet_meshes(self, shapenet_catids, shapenet_ids):
        # load simplified gt meshes (for evaluation)
        shape_data_list = []
        for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids):
            mesh_file = os.path.join(self.shapenet_path, 'watertight_scaled_simplified', shapenet_catid, shapenet_id + '.off')
            mesh = trimesh.load(mesh_file, process=False)
            shape_data_list.append(mesh)
        return shape_data_list

    def get_shapenet_points(self, shapenet_catids, shapenet_ids, transform=None):
        '''Load points and corresponding occ values.'''
        shape_data_list = []
        for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids):
            points_dict = np.load(os.path.join(self.shapenet_path, 'point', shapenet_catid, shapenet_id + '.npz'))
            points = points_dict['points']
            occupancies = points_dict['occupancies']

            # # Break symmetry if given in float16:
            # if points.dtype == np.float16 and self.mode == 'train':
            #     points = points.astype(np.float32)
            #     points += 1e-4 * np.random.randn(*points.shape)
            # else:
            #     points = points.astype(np.float32)
            # if self.points_unpackbits:
            #     occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            # occupancies = occupancies.astype(np.float32)

            data = {'points':points, 'occ': occupancies}

            shape_data_list.append(data)

        return recursive_cat_to_numpy(shape_data_list) # {points: [M, N, 3], occ: [M, N]}

    def random_sampling(self, pc, num_sample, replace=None, return_choices=False):
        """ Input is NxC, output is num_samplexC
        """
        if replace is None: 
            replace = (pc.shape[0]<num_sample)
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
        if return_choices:
            return pc[choices], choices
        else:
            return pc[choices]

# collate instances
def recursive_cat_to_numpy(data_list):
    '''Covert a list of dict to dict of numpy arrays.'''
    out_dict = {}
    for key, value in data_list[0].items():
        if key in ['full_points']:
            out_dict = {**out_dict, key: [data[key] for data in data_list]}  
        elif isinstance(value, np.ndarray):
            out_dict = {**out_dict, key: np.concatenate([data[key][np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, dict):
            out_dict =  {**out_dict, **recursive_cat_to_numpy(value)}
        elif np.isscalar(value):
            out_dict = {**out_dict, key: np.concatenate([np.array([data[key]])[np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, list):
            out_dict = {**out_dict, key: np.concatenate([np.array(data[key])[np.newaxis] for data in data_list], axis=0)}
    return out_dict

def collate_fn(batch):
    # batch must be a list of dict
    collated_batch = {}
    for key in batch[0]:
        # do not covert unpadded list to Tensor
        if key in ['object_meshes']:
            collated_batch[key] = [sample[key] for sample in batch] 
        # # sparse tensor
        # elif isinstance(batch[0][key], SparseTensor):
        #     collated_batch[key] = sparse_collate_tensors([sample[key] for sample in batch])
        # else (default behaviour)
        elif isinstance(batch[0][key], np.ndarray):
            #print(key, [sample[key].shape for sample in batch])
            collated_batch[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch], axis=0)
        elif isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([sample[key] for sample in batch], axis=0)
        elif isinstance(batch[0][key], dict):
            collated_batch[key] = collate_fn([sample[key] for sample in batch])
        else:
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataloader(cfg, mode='train', test_split=None):
    
    if mode == 'train':
        dataset = ShapeNetDataset(cfg, mode)
    elif mode == 'val':
        dataset = ShapeNetDataset(cfg, mode)
    # use scannet to test
    else:
        dataset = ScanNetDataset(cfg, mode, test_split)

    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.num_workers,
                            batch_size=(cfg.train_batch if mode == 'train' else cfg.test_batch),
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    return dataloader
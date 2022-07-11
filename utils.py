import os
import glob
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
from skimage import io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from apex import amp

from collections import defaultdict

import trimesh

CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']

Z = 128

class AttributeDict(dict):
    def __init__(self, d=None):
        #super().__init__(AttributeDict)
        if d is not None:
            for k, v in d.items():
                self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g. 
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class PolyMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = faces
    
    def export(self, name):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(self.vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(self.faces))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(self.vertices)):
            fout.write(str(self.vertices[ii][0])+" "+str(self.vertices[ii][1])+" "+str(self.vertices[ii][2])+"\n")
        for ii in range(len(self.faces)):
            fout.write(str(len(self.faces[ii])))
            for jj in range(len(self.faces[ii])):
                fout.write(" "+str(self.faces[ii][jj]))
            fout.write("\n")
        fout.close()

    def to_trimesh(self):
        # triangulate polygons
        triangles = []
        # each face contains 3 / 4 points
        for f in self.faces:
            if len(f) == 3:
                triangles.append(f)
            else:
                assert len(f) == 4
                # silly way, use 3 triangles, but the optimal is 2 triangles. lazy to detect the diagonal. should have no bad effect.
                triangles.append(f[[0,1,2]])
                triangles.append(f[[0,1,3]])
                triangles.append(f[[0,2,3]])

        return trimesh.Trimesh(self.vertices, np.array(triangles), process=False)


class RMSEMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, data, preds, truths, eval=False):
        preds, truths = self.prepare_inputs(preds, truths) # [B, 1, H, W]

        # rmse
        rmse = np.sqrt(np.mean(np.power(preds - truths, 2)))

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "rmse"), self.measure(), global_step)

    def report(self):
        return f'RMSE = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 args,
                 name, # name of this experiment
                 model, # network 
                 objective=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 lr_scheduler=None, # scheduler
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 opt_level='O0', # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.args = args
        self.name = name
        self.mute = mute
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.opt_level = opt_level
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.log_ptr = None

        self.model.to(self.device)
        if isinstance(self.objective, nn.Module):
            self.objective.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level, verbosity=0)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Model randomly initialized ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args):
        if self.local_rank == 0:
            if not self.mute: 
                print(*args)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)    

    ### ------------------------------	

    def get_phase(self):
        if self.epoch <= self.args.phase_0_epoch:
            phase = 0
        elif self.epoch <= self.args.phase_1_epoch:
            phase = 1
        else:
            phase = 2
        return phase


    def train_step(self, data):
        # change phase 
        phase = self.get_phase()
        data['epoch'] = self.epoch # beta annealing
        loss_dict = self.model(data, phase=phase, return_loss=True, return_mesh=False)['loss']
        return loss_dict

    def eval_step(self, data):
        # only call decoder and generator
        phase = self.get_phase()
        meshes = self.model(data, phase=phase, return_loss=False, return_mesh=True, generate=True)['meshes']
        return meshes

    def test_step(self, data):  
        # change phase 
        phase = self.get_phase()
        meshes = self.model(data, phase=phase, return_loss=False, return_mesh=True)['meshes']
        return meshes

    def test_z_step(self, data):  
        # only call encoder, so no matter with phase
        zs = self.model(data, phase=0, return_loss=False, return_mesh=False, return_z=True)['zs']
        return zs

    ### ------------------------------

    def train(self, train_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, num_samples):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        save_path = os.path.join(self.workspace, 'results', f'{self.name}', f'evaluate')
        self.evaluate_one_epoch(num_samples, save_path)
        self.use_tensorboardX = use_tensorboardX

    def get_loss_description(self, loss_dict):
        res = '|'
        for k, v in loss_dict.items():
            res += f' {k}={v.item():.8f} |'
        return res

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")

        loss_history = {}

        self.model.train()
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)
            loss_dict = self.train_step(data)

            with amp.scale_loss(loss_dict['loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            
            for loss_name, loss_tensor in loss_dict.items():
                if loss_name not in loss_history:
                    loss_history[loss_name] = [loss_tensor.item()]
                else:
                    loss_history[loss_name].append(loss_tensor.item())

            if self.local_rank == 0:
                        
                if self.use_tensorboardX:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(self.get_loss_description(loss_dict) + f" lr={self.optimizer.param_groups[0]['lr']} |")
                else:
                    pbar.set_description(self.get_loss_description(loss_dict))

                pbar.update(loader.batch_size * self.world_size)

        avg_loss = {}
        for loss_name, loss_list in loss_history.items():
            avg_loss[loss_name] = np.mean(loss_list)
            
        self.stats['loss'].append(avg_loss)

        if self.local_rank == 0:
            pbar.close()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(avg_loss['loss'])
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch} " + self.get_loss_description(avg_loss))

    # generate & save random samples.
    def evaluate_one_epoch(self, num_samples=8, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results', f'{self.name}', f'epoch_{self.epoch}')
        os.makedirs(save_path, exist_ok=True)
        self.log(f"++> Evaluate at epoch {self.epoch}, save to {save_path} ...")
        self.model.eval()
        with torch.no_grad():

            
            batch_size = 16
            for start in range(0, num_samples, batch_size):

                zs = torch.randn(batch_size, Z) # [B, 256]
                labels = (torch.arange(batch_size) + start) % self.args.num_classes # [B, ]

                # random samples
                data = {}
                data['zs'] = zs
                data['labels'] = labels
                data = self.prepare_data(data)
                print(data['zs'].shape, data['labels'].shape)

                meshes = self.eval_step(data) # [B, M], trimesh object, padded
                
                # per batch, per mesh
                B = len(meshes)
                labels = data['labels']
                for b in range(B):
                    mesh = meshes[b]
                    mesh.export(os.path.join(save_path, f'{CAD_labels[labels[b].item()]}_{b+start}.ply'))
                    print(f'save to {CAD_labels[labels[b].item()]}_{b+start}.ply')


        self.log(f"++> Evaluate epoch {self.epoch} Finished ")


    # interpolated generate
    def interpolated_generate(self, num_samples=8, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results', f'{self.name}', f'interpolated')
        os.makedirs(save_path, exist_ok=True)
        self.log(f"++> Generate at epoch {self.epoch}, save to {save_path} ...")
        self.model.eval()
        with torch.no_grad():
            
            # random start and end
            z0 = torch.randn(Z) # [256]
            z1 = torch.randn(Z) # [256]
            zs = torch.stack([t * z0 + (1 - t) * z1 for t in np.linspace(0, 1, num_samples)], dim=0) # [N, 256]

            for cls in range(self.args.num_classes):
                labels = torch.LongTensor([cls] * num_samples)
                
                data = {}
                data['zs'] = zs
                data['labels'] = labels
                data = self.prepare_data(data)
                print(data['zs'].shape, data['labels'].shape)

                meshes = self.eval_step(data) # [B, M], trimesh object, padded
                
                # per batch, per mesh
                B = len(meshes)
                labels = data['labels']
                for b in range(B):
                    mesh = meshes[b]
                    mesh.export(os.path.join(save_path, f'{CAD_labels[labels[b].item()]}_{b}.ply'))
                    print(f'save to {CAD_labels[labels[b].item()]}_{b}.ply')


        self.log(f"++> Generate at epoch {self.epoch} Finished ")

    # test on shapenet (auto-encoder)
    def test_shapenet(self, loader, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results', f'{self.name}', 'shapenet')
        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                
                data = self.prepare_data(data)
                meshes = self.test_step(data) # [B, M], trimesh object, padded
                labels = data['labels']
                name = data['name']

                # per batch, per mesh
                B = len(meshes)
                for b in range(B):
                    mesh = meshes[b]
                    mesh.export(os.path.join(save_path, f'{CAD_labels[labels[b].item()]}_{name[b]}.ply'))
                    print(f'save to {CAD_labels[labels[b].item()]}_{name[b]}.ply')

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    def test_shapenet_z(self, loader):
        
        self.log(f"==> Start test shapenet z")

        # SAVE IN MEM: zs [N, C], labels [N, 1], GT name [N,]
        # use: first select by label, then calc dist with zs, then select the nearest z/name.
        # save three pickled arrays
        ZS = []
        LBLS = []
        NAMES = []

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                
                data = self.prepare_data(data)
                labels = data['labels'] # [B]
                names = data['name'] # [B]
                zs = self.test_z_step(data) # [B, C]

                labels = labels.detach().cpu().numpy()
                zs = zs.detach().cpu().numpy()

                ZS.append(zs)
                LBLS.append(labels)
                NAMES.append(names)

                pbar.update(loader.batch_size)

        ZS = np.concatenate(ZS, axis=0) # [N, C]
        LBLS = np.concatenate(LBLS, axis=0)
        NAMES = np.concatenate(NAMES, axis=0)
        
        filename = os.path.join(self.workspace, 'database.npz')
        np.savez(filename, zs=ZS, lbls=LBLS, names=NAMES)

        self.log(f"==> Finished Test.")

    def test_scannet_z(self, loader):
        
        self.log(f"==> Start test shapenet z")

        # SAVE IN MEM: zs [N, C], labels [N, 1], GT name [N,]
        # use: first select by label, then calc dist with zs, then select the nearest z/name.
        # save three pickled arrays
        ZS = []
        LBLS = []
        NAMES = []
        vis = set()

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                
                data = self.prepare_data(data)
                zs = self.test_z_step(data) # [B, C]

                scan_names = data['scan_name'] # [B, ]
                valid_mask = data['mask'] # [B, M]
                labels = data['labels'].long() # [B, M]
                shapenet_catids = data['shapenet_catids']
                shapenet_ids = data['shapenet_ids'] # no .npz
                cnts = data['cnt'] # [B]

                labels = labels.detach().cpu().numpy()
                zs = zs.detach().cpu().numpy()

                B = len(cnts)
                for b in range(B):
                    for i in range(cnts[b]):
                        z = zs[b][i] # remove pad
                        lbl = labels[b][i]
                        cat_name = shapenet_catids[b][i]
                        name = shapenet_ids[b][i] + '.npz'

                        if cat_name + name not in vis:
                            vis.add(cat_name + name)
                            print(cat_name + name, z.shape, lbl)

                            ZS.append(z)
                            LBLS.append(lbl)
                            NAMES.append(name)

                pbar.update(loader.batch_size)

        ZS = np.stack(ZS, axis=0) # [N, C]
        LBLS = np.stack(LBLS, axis=0)
        NAMES = np.stack(NAMES, axis=0)
        
        filename = os.path.join(self.workspace, 'database_scannet.npz')
        np.savez(filename, zs=ZS, lbls=LBLS, names=NAMES)

        self.log(f"==> Finished generating database_scannet. {ZS.shape}, {LBLS.shape}")

    # test on scannet
    def test(self, loader, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results', f'{self.name}', 'scannet')
        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                
                # assert single batch !!! so B == 1, and M is the actual B.
                
                data = self.prepare_data(data)
                meshes = self.test_step(data) # [B, M], trimesh object, padded

                scan_names = data['scan_name'] # [B, ]
                valid_mask = data['mask'] # [B, M]
                labels = data['labels'].long() # [B, M]

                # per batch, per mesh
                B, M = valid_mask.shape
                for b in range(B):
                    for i in range(M):
                        if valid_mask[b, i] == 0: # meshes[b, i] is not None
                            continue
                        mesh = meshes[b][i]
                        mesh.export(os.path.join(save_path, f'{scan_names[b]}_{i}_{CAD_labels[labels[b, i].item()]}.ply'))
                        print(f'save to {scan_names[b]}_{i}_{CAD_labels[labels[b, i].item()]}.ply')

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    # save scannet z vectors
    def test_z(self, loader, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results_z', f'{self.name}')
        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test (Save z), save results to {save_path}")

        # results_z/name/scene0100_01/zs.npz

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                
                data = self.prepare_data(data)
                zs = self.test_z_step(data) # [B, M, C], z_vectors
                zs = zs.detach().cpu().numpy()

                scan_names = data['scan_name'] # [B, ]
                valid_mask = data['mask'] # [B, M]
                labels = data['labels'].long() # [B, M]
                cnts = data['cnt'] # [B]

                B = len(cnts)
                for b in range(B):
                    z = zs[b][:cnts[b]] # remove pad
                    os.makedirs(os.path.join(save_path, scan_names[b]), exist_ok=True)
                    np.savez(os.path.join(save_path, scan_names[b], 'zs_vae1.npz'), zs=z)
                    print(f"save {z.shape} to {os.path.join(save_path, scan_names[b], 'zs_vae1.npz')}")

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")        

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
            'model': self.model.state_dict(),
        }

        if full:
            state['amp'] = amp.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        if not best:

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        self.model.load_state_dict(checkpoint_dict['model'])
        self.log("[INFO] loaded model.")

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch'] + 1
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self.log("[INFO] loaded optimizer.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
                

        if 'amp' in checkpoint_dict:
            amp.load_state_dict(checkpoint_dict['amp'])
            self.log("[INFO] loaded amp.")
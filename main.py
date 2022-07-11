import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from dataloader import get_dataloader
from models import get_model

import pprint
import argparse

args = AttributeDict({
    'name': 'rfs_bsp',
    'seed': 0,
    # dataloader
    'data_root': './',
    'train_batch': 16,
    'test_batch': 1, 
    'num_workers': 8,
    'num_point': 120000, # RfD is 8w, pg is 25w, mean seems to be 20w~
    'num_classes': 8,
    'use_color': False,
    'no_height': True,
    'points_unpackbits': True,
    'points_subsample': [2048, 2048],
    # optimize
    'epoch': 900, 
    'eval_interval': 10, 
    'checkpoint': 'latest',
    'lr': 1e-4,
    'lr_step': 200, 
    'lr_gamma': 0.5,
    'test': False,
    'save': False,
    # eval
    'mise_resolution_0': 32, 
    'mise_upsampling_steps': 0, 
    'mise_threshold': 0.5, 
    'fast_eval': False, # only use first 50 to eval
    # bsp-cvae
    'voxel_resolution': [64, 64, 64],
    'num_planes': 4096,
    'num_convexes': 256,
    'num_feats': 32,
    'mesh_gen': 'bspt', # 'bspt' or 'mcubes'
    'phase_0_epoch': 400,
    'phase_1_epoch': 800,
    'beta': 0.1,
    'beta_max_epoch': 600,
})

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test_shapenet', action='store_true')
parser.add_argument('--test_scannet', action='store_true')
parser.add_argument('--save_z', action='store_true')
parser.add_argument('--save_db', action='store_true')
parser.add_argument('--generate', action='store_true')
parser.add_argument('--interpolated_generate', action='store_true')
parser.add_argument('--checkpoint', type=str, default='latest')
args2 = parser.parse_args()

args.update(args2.__dict__)

pprint.pprint(args)

# seed
seed_everything(args.seed)

# model
model = get_model(args)

# trainer
if args.train:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, optimizer=optimizer, lr_scheduler=scheduler, device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval)
else:
    trainer = Trainer(args, args.name, model, device='cuda', use_checkpoint=args.checkpoint)

# train
if args.train:
    train_loader = get_dataloader(args, 'train')
    trainer.train(train_loader, args.epoch)

# test
if args.test_shapenet:
    test_loader = get_dataloader(args, 'val') # val == shapenet
    trainer.test_shapenet(test_loader)
if args.test_scannet:
    test_loader = get_dataloader(args, 'test', 'val') # test == scannet val
    trainer.test(test_loader)

# generate
if args.generate:
    trainer.evaluate(num_samples=800)

# interpolated generate
if args.interpolated_generate:
    trainer.interpolated_generate(num_samples=8)

# save z
if args.save_z:
    train_loader = get_dataloader(args, 'test', 'train') # scannet train
    test_loader = get_dataloader(args, 'test', 'val') # scannet val
    trainer.test_z(train_loader)
    trainer.test_z(test_loader)

# save db
if args.save_db:
    #test_loader = get_dataloader(args, 'val') # val == shapenet
    #trainer.test_shapenet_z(test_loader)

    test_loader2 = get_dataloader(args, 'test', 'train') # scannet train
    trainer.test_scannet_z(test_loader2)

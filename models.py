import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_coord

import mcubes
import trimesh

from external.libmise import MISE

from external.bspt import get_mesh_watertight
#from external.bspt_slow import get_mesh_watertight # for debug

from utils import PolyMesh

Z = 128


class generator(nn.Module):
    def __init__(self, p_dim, c_dim):
        super(generator, self).__init__()
        
        self.p_dim = p_dim
        self.c_dim = c_dim
        
        # same for all observations.
        self.convex_layer_weights = nn.Parameter(torch.zeros((self.p_dim, self.c_dim)))
        self.concave_layer_weights = nn.Parameter(torch.zeros((self.c_dim, 1)))

        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

    def forward(self, points, planes, phase=0):
        # 0: continuous pretraining (S+)
        # 1: discrete (S*)
        # 2: discrete + overlap loss (S*)
        # 3: soft discrete
        # 4: soft discrete + overlap loss

        # to homogeneous coordinates
        if points.shape[-1] == 3:
            points = torch.cat([points, torch.ones(points.shape[0], points.shape[1], 1).to(points.device)], dim=-1)

        #level 1
        h1 = torch.matmul(points, planes) # [B, N, 4] x [B, 4, P] -> [B, N, P], if the point is in the correct side of the plane
        h1 = torch.clamp(h1, min=0)

        if phase == 0:

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights) # [B, N, C], if the point is in the convex
            h2 = torch.clamp(1-h2, min=0, max=1)

            #level 3
            h3 = torch.matmul(h2, self.concave_layer_weights) # [B, N, 1], if the point is in the concave (final shape)
            h3 = torch.clamp(h3, min=0, max=1)

            return h2, h3
        elif phase == 1 or phase == 2:

            #level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights > 0.01).float())

            #level 3
            h3 = torch.min(h2, dim=2, keepdim=True)[0]

            return h2, h3


class encoder(nn.Module):
    def __init__(self, ef_dim, in_dim=1):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(in_dim, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, Z * 2, 4, stride=1, padding=0, bias=True)
        self.bn_1 = nn.BatchNorm3d(self.ef_dim)
        self.bn_2 = nn.BatchNorm3d(self.ef_dim*2)
        self.bn_3 = nn.BatchNorm3d(self.ef_dim*4)
        self.bn_4 = nn.BatchNorm3d(self.ef_dim*8)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias,0)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias,0)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias,0)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias,0)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)

    def forward(self, inputs):
        d_1 = self.conv_1(inputs)
        d_1 = self.bn_1(d_1)
        d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

        d_2 = self.conv_2(d_1)
        d_2 = self.bn_2(d_2)
        d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)
        
        d_3 = self.conv_3(d_2)
        d_3 = self.bn_3(d_3)
        d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

        d_4 = self.conv_4(d_3)
        d_4 = self.bn_4(d_4)
        d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

        d_5 = self.conv_5(d_4) # if input is [B, Cin, 64, 64, 64], d_5 is exatly [B, Cout, 1, 1, 1]

        d_5 = d_5.view(-1, Z * 2) # [B, 256*2], 256 = mu, 256 = logvar

        return d_5


class decoder(nn.Module):
    def __init__(self, ef_dim, num_classes, p_dim):
        super(decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.num_classes = num_classes
        self.linear_1 = nn.Linear(Z + self.num_classes, self.ef_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
        self.bn_1 = nn.BatchNorm1d(self.ef_dim*16)
        self.bn_2 = nn.BatchNorm1d(self.ef_dim*32)
        self.bn_3 = nn.BatchNorm1d(self.ef_dim*64)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias,0)


    def forward(self, zs_labels):

        l1 = self.linear_1(zs_labels)
        l1 = self.bn_1(l1)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = self.bn_2(l2)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = self.bn_3(l3)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4


def reparametrize(mu, log_sigma):
    stddev = 0.5 * torch.exp(log_sigma)
    basis = torch.randn(mu.shape).to(mu.device)
    return mu + basis * stddev

def KL_loss(mu, log_sigma):
    # mu, log_sigma: [B, fout]
    loss = -0.5 * torch.mean(1 + log_sigma - torch.exp(log_sigma) - mu**2)
    return loss

class BSP_CVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.encoder = encoder(args.num_feats, (4 if args.no_height else 5) + args.num_classes)
        self.decoder = decoder(args.num_feats, args.num_classes, args.num_planes)
        self.generator = generator(args.num_planes, args.num_convexes)
    
    # loss
    def calculate_loss(self, convexes, pred_occs, occs, W_cvx, W_aux, feats, epoch):
        # convexes - network output (convex layer), the last dim is the number of convexes
        # pred_occs - network output (final output), [B, 1, N]
        # occs - ground truth inside-outside value for each point, [B, 1, N]
        # W_cvx - connections T
        # W_aux - auxiliary weights W

        B = occs.shape[0]
        pred_occs = pred_occs.view(B, -1)
        occs = occs.view(B, -1)

        mu = feats[:, :Z]
        log_sigma = feats[:, Z:]
        loss_kl = KL_loss(mu, log_sigma)

        # beta-annealing
        beta = self.args.beta * (epoch / self.args.beta_max_epoch)

        loss_dict = {}
        loss_dict['loss_kl'] = loss_kl

        if self.phase == 0:
            # phase 0 continuous for better convergence
            # L_recon + L_W + L_T
            loss_sp = torch.mean((occs - pred_occs)**2) 
            loss = loss_sp + torch.sum(torch.abs(W_aux-1)) + (torch.sum(torch.clamp(W_cvx-1, min=0) - torch.clamp(W_cvx, max=0))) + beta * loss_kl
            loss_dict['loss_sp'] = loss_sp
            
        elif self.phase == 1:
            # phase 1 hard discrete for bsp
            # L_recon
            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)))
            loss = loss_sp + beta * loss_kl
            loss_dict['loss_sp'] = loss_sp

        elif self.phase == 2:
            # phase 2 hard discrete for bsp with L_overlap
            # L_recon + L_overlap
            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)))

            G2_inside = (convexes < 0.01).float()
            bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True)>1).float()
            loss_overlap = -torch.mean(convexes*occs.unsqueeze(-1)*bmask) 

            loss = loss_sp + loss_overlap + beta * loss_kl
            loss_dict['loss_sp'] = loss_sp
            loss_dict['loss_overlap'] = loss_overlap
            
        loss_dict['loss'] = loss

        return loss_dict

    def forward(self, data, phase=0, return_loss=True, return_mesh=False, return_z=False, generate=False):

        # set phase
        self.phase = phase

        is_scannet = False
        
        # enc + dec
        if not generate:
            # extract data
            voxels = data['voxels'] # [B, C, H, W, D] or [B, M, C, H, W, D]
            labels = data['labels'] # [B, ] or [B, M]

            if len(voxels.shape) == 6: 
                is_scannet = True
                mask = data['mask']
                bbox = data['bbox'] # [B_, M, 6]
                B_, M, C, H, W, D = voxels.shape    
                voxels = voxels.view(-1, C, H, W, D)
                labels = labels.view(-1)
                mask = mask.view(-1)
                bbox = bbox.view(-1, 7)
                B = B_ * M
            else:
                B, C, H, W, D = voxels.shape
                bbox = [None] * B

            labels = F.one_hot(labels, self.args.num_classes) # [B, 8]

            voxels_labels = torch.cat([voxels, labels.view(B,-1,1,1,1).expand(B,-1,H,W,D)], dim=1)
            feats = self.encoder(voxels_labels) # [B, C]
            
            if self.training:
                zs = reparametrize(feats[:, :Z], feats[:, Z:]) # [B, 256]
            else:
                zs = feats[:, :Z]

        # just dec
        else:
            labels = data['labels'] # [B, ]
            B = labels.shape[0]
            bbox = [None] * B

            labels = F.one_hot(labels, self.args.num_classes) # [B, 8]

            zs = data['zs'] # [B, 256]
        
        zs_labels = torch.cat([zs, labels], dim=1) # [B, 256+8]
        planes = self.decoder(zs_labels) # [B, 4, P]

        res = {}

        # save z before reparam.
        if return_z:
            if is_scannet:
                zs = feats.view(B_, M, -1)
            else:
                zs = feats
            res['zs'] = zs

        if return_mesh:
            # loop each proposal and extract mesh 
            meshes = []
            for i in range(B):
                
                #print(f'[INFO] generating meshes... {i} / {B*M}')

                # if invalid
                if is_scannet and mask[i] == 0:
                    meshes.append(None)
                    continue
                
                if self.args.mesh_gen == 'mcubes' or self.phase == 0:

                    if self.phase == 0: # 1 means occ
                        threshold = 0.5
                        voxels = self.implicit_to_voxels(planes[i], self.args.mise_resolution_0, self.args.mise_upsampling_steps, threshold) # [H, W, D]

                    elif self.phase >= 1: # 0 means occ
                        # do not support mise
                        assert self.args.mise_upsampling_steps == 0
                        threshold = 0.99
                        voxels = self.implicit_to_voxels(planes[i], self.args.mise_resolution_0, self.args.mise_upsampling_steps, threshold) # [H, W, D]
                        voxels = 1 - voxels

                    mesh = self.voxels_to_mesh(voxels, threshold=threshold, bbox=bbox[i])

                elif self.args.mesh_gen == 'bspt':
                    # only support phase 1
                    assert self.phase >= 1 
                    bsp = self.implicit_to_bsp(planes[i], self.args.mise_resolution_0)
                    mesh = self.bsp_to_mesh(bsp, planes[i], bbox=bbox[i])

                else:
                    raise NotImplementedError(self.args.mesh_gen)

                meshes.append(mesh)

            if is_scannet:
                meshes = [meshes[i:i+M] for i in range(0, B_*M, M)]

            res['meshes'] = meshes

        if return_loss:
            points = data['points'] # [B, N, 3], world coord !
            occs = data['occ'] # [B, N]

            # extract gt for input
            B, N, _ = points.shape
            
            # query
            convexes, occs_pred = self.generator(points, planes, self.phase)
            
            loss = self.calculate_loss(convexes, occs_pred, occs, self.generator.convex_layer_weights, self.generator.concave_layer_weights, feats, data['epoch'])

            res['loss'] = loss
        
        return res
    
    # use libmise, only support cubic query.
    def implicit_to_voxels(self, planes, resolution_0=16, upsampling_steps=3, threshold=0):
        # planes: [4, P], assert batch == 1
        # resolution_0: for MISE, initial voxel res
        # upsampling_steps: for MISE, upsampling ratio (2^x)
        # threshold: for MISE, determine whether a voxel is occupied (active). should be 0 if we use logits.

        # naive dense query
        if upsampling_steps == 0:
            points = make_coord([resolution_0]*3).to(planes.device) # [HWD, 3]        
            
            _, voxels = self.generator(points.unsqueeze(0), planes.unsqueeze(0), phase=self.phase)
            voxels = voxels.view([resolution_0]*3) # [1, 1, N] --> [H, W, D]

        # use MISE for sparse query, should be faster
        else:
            mise = MISE(resolution_0, upsampling_steps, threshold)
            point_coords = mise.query() # [(H+1)^3, 3], mise assume values on grid corners, but we assume values on grid center.
            
            while point_coords.shape[0] != 0:

                # normalize to [-1, 1]
                half_grid_length = 1 / mise.resolution # [1], assume cubic

                # for coords, (max_xyz - min_xyz) == (resolution - 1), e.g., [0, 1, 2, 3] --> mx - mn = 3, res = 4
                points = (2 * point_coords / (mise.resolution - 1) - 1) * (1 - half_grid_length) # [0, H] in [-1, 1], [H+1] > 1

                # to tensor
                points = torch.FloatTensor(points).to(planes.device)
                
                # evaluate model and update
                _, values = self.generator(points.unsqueeze(0), planes.unsqueeze(0), phase=self.phase)
                values = values.view(-1)
                values = values.detach().cpu().numpy().astype(np.float64)

                mise.update(point_coords, values)
                point_coords = mise.query()

            voxels = mise.to_dense()[:-1, :-1, :-1] # [H+1, W+1, D+1] --> [H, W, D], grid corner --> grid center
        
        return voxels

    # query convexes, only support naive dense query
    def implicit_to_bsp(self, planes, resolution_0=64):
    
        points = make_coord([resolution_0]*3).to(planes.device) # [HWD, 3]        
        
        # phase must be 1, so 0 == occ
        bsp, _ = self.generator(points.unsqueeze(0), planes.unsqueeze(0), phase=1)
        bsp = bsp.view([resolution_0]*3 + [bsp.shape[-1]]) # [1, N, C] --> [H, W, D, C]
        
        return bsp

    # use bsp-tree to extract mesh
    def bsp_to_mesh(self, bsp, planes, bbox=None):
        # bsp: [H, W, D, C]
        # planes: [4, P]

        if bbox is not None and torch.is_tensor(bbox):
            bbox = bbox.detach().cpu().numpy()

        # [cvx1, cvx2, ...], cvx1 = [[a1,b1,c1,d1], ...]
        bsp_convexes = []
        
        if torch.is_tensor(bsp):
            bsp = bsp.detach().cpu().numpy()
        
        # if the point is inside the convex.
        bsp = (bsp < 0.01).astype(np.int32)
        bsp_sum = bsp.sum(axis=3) # [H, W, D]
        
        W_cvx = self.generator.convex_layer_weights.detach().cpu().numpy() # [P, C]

        for i in range(self.args.num_convexes):
            cvx = bsp[:,:,:,i]
            if np.max(cvx) > 0: # if at least one voxel is inside this convex
                # if this convex is redundant, i.e. the convex is inside the shape
                if np.min(bsp_sum - cvx * 2) >= 0:
                    bsp_sum = bsp_sum - cvx
                # add convex (by finding planes composing this convex)
                else:
                    cvx_planes = []
                    for j in range(self.args.num_planes):
                        # the j-th plane is part of the i-th convex.
                        if W_cvx[j, i] > 0.01:
                            cvx_planes.append((-planes[:, j]).tolist())
                    if len(cvx_planes) > 0:
                        bsp_convexes.append(np.array(cvx_planes, np.float32))

        
        vertices, polygons = get_mesh_watertight(bsp_convexes) # not necessarily a tri-angular mesh!
        vertices = np.array(vertices)
        
        # vertices already normalized to [-1, 1] ! determined by the border_limit in bspt.

        # fit back into world bbox if possible
        if bbox is not None:
            # non-oriented 6D bbox
            if bbox.shape[0] == 6:
                min_xyz, max_xyz = bbox[:3], bbox[3:]
                half_grid_length_input = 1 / np.array(self.args.voxel_resolution) # [3]
                vertices = (vertices / (1 - half_grid_length_input) + 1) * (max_xyz - min_xyz + 1e-5) / 2 + min_xyz
            # oriented 7D bbox
            elif bbox.shape[0] == 7:
                print(bbox)
                center, bsize, orientation = bbox[:3], bbox[3:6], bbox[6]
                vertices = (vertices / 2) * bsize
                axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
                vertices = vertices.dot(axis_rectified)
                vertices = vertices + center


        # create mesh
        mesh = PolyMesh(vertices, polygons)

        return mesh


    # use mcubes to extract mesh
    def voxels_to_mesh(self, voxels, bbox=None, threshold=0.5):
        # voxels: [H, W, D]
        # bbox: [6], min_xyz, max_xyz

        # to numpy
        if torch.is_tensor(voxels):
            voxels = voxels.detach().cpu().numpy()
        if bbox is not None and torch.is_tensor(bbox):
            bbox = bbox.detach().cpu().numpy()
        
        # make sure that mesh is watertight (by padding non-occupied borders)
        voxels_padded = np.pad(voxels, 1, 'constant', constant_values=-1e6) # logits, -1e6 --sigmoid--> 0

        # mcubes assumes voxel values are on grid center, so no problem !
        vertices, triangles = mcubes.marching_cubes(voxels_padded, threshold)

        # undo padding
        vertices -= 1 

        # normalize to [-1, 1]
        resolution = np.array(voxels.shape)
        half_grid_length = 1 / resolution # [3]
        vertices = (2 * vertices / (resolution - 1) - 1) * (1 - half_grid_length)

        # fit back into world bbox if possible
        if bbox is not None:
            if bbox.shape[0] == 6:
                min_xyz, max_xyz = bbox[:3], bbox[3:]
                half_grid_length_input = 1 / np.array(self.args.voxel_resolution) # [3]
                vertices = (vertices / (1 - half_grid_length_input) + 1) * (max_xyz - min_xyz + 1e-5) / 2 + min_xyz
            elif bbox.shape[0] == 7:
                center, bsize, orientation = bbox[:3], bbox[3:6], bbox[6]
                vertices = (vertices / 2) * bsize
                axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
                vertices = vertices.dot(axis_rectified)
                vertices = vertices + center


        # create mesh
        mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)

        return mesh


def get_model(cfg):
    return BSP_CVAE(cfg)
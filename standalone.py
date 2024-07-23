import os
import cv2
import pickle
import yaml
import json
import torch
import random
import numpy as np
import pickle as pkl
from typing import Tuple
import imageio.v2 as imageio
from easydict import EasyDict
from easydict import EasyDict
from src.data.components.amelia_dataset import AmeliaDataset
from src.models.components.amelia import AmeliaTF # Context aware model
from src.models.components.amelia_traj import AmeliaTraj # Non context aware model
import src.utils.global_masks as G
from geographiclib.geodesic import Geodesic
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

class SocialTrajPred():
    def __init__(self, airport: str, model, dataloader: AmeliaDataset,
                 use_map: bool = True):

        # Create output directory
        self.AIRPORT = airport

        # Load model and dataloaders
        self.model = model
        self.dataloader = dataloader
        self.load_assets()
        
        # Configure CUDA
        torch.set_printoptions(precision=10, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
        torch.cuda.set_device(0)
        self.model.cuda()
    
    def load_assets(self):
        # Init attributes
        self.dataloader.out_dirs = {}
        self.dataloader.semantic_maps = {}
        self.dataloader.semantic_pkl = {}
        self.dataloader.limits = {}
        self.dataloader.ref_data = {}
        self.dataloader.scenario_list = {}
        self.dataloader.hold_lines = {}
        self.dataloader.data_files = []
         
        graph_file = os.path.join(self.dataloader.context_dir, self.AIRPORT, 'semantic_graph.pkl')
        with open(graph_file, 'rb') as f:
            temp_dict = pickle.load(f)
            self.dataloader.semantic_pkl[self.AIRPORT] = temp_dict
            self.dataloader.semantic_maps[self.AIRPORT] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
            self.dataloader.hold_lines[self.AIRPORT] = temp_dict['hold_lines']
            
        limits_file = os.path.join(self.dataloader.assets_dir, self.AIRPORT, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.dataloader.ref_data[self.AIRPORT] = EasyDict(json.load(fp))

        self.dataloader.limits[self.AIRPORT] = (
            self.dataloader.ref_data[self.AIRPORT].espg_4326.north, 
            self.dataloader.ref_data[self.AIRPORT].espg_4326.east, 
            self.dataloader.ref_data[self.AIRPORT].espg_4326.south, 
            self.dataloader.ref_data[self.AIRPORT].espg_4326.west
        )
        
    def dict_to_tensor(self, dict):
        tensor_dict = {}
        for key, value in dict.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value).to(device='cuda')
        return tensor_dict
            
    def load_ckpt(self,ckpt_path: str, from_pickle: bool = False):
        """
        Converts pytorch lightning state dict to torch state dict by removing net. prefix and load this to the GPT module.
        """
        if(from_pickle):
            with open(ckpt_path,'rb') as file:
                state_dict = pkl.load(file)
        else:
            checkpoint = torch.load(ckpt_path,  map_location=torch.device('cuda:0'))
            state_dict = checkpoint['state_dict']
            state_dict = {k.partition('net.')[2]: v for k,v in state_dict.items()}
        self.model.load_state_dict(state_dict) 
          
    def forward(self, scene_data, random_ego = False) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # Transfrom scene in local frame
        transformed_scene = self.dataloader.transform_scene_data(scene_data, random_ego = random_ego)
        transformed_scene = self.dataloader.collate_batch([transformed_scene])
        # Prepare inputs
        Y = transformed_scene['scene_dict']['rel_sequences'].to(device="cuda")
        X = torch.zeros_like(Y).type(torch.float).to(device="cuda")
        X[:, :, :self.dataloader.hist_len] = Y[:, :, :self.dataloader.hist_len]
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]] 
        context = transformed_scene['scene_dict']['context'].to(device="cuda")
        adjacency = transformed_scene['scene_dict']['adjacency'].to(device="cuda")
        ego_agent = transformed_scene['scene_dict']['ego_agent_id']
        masks = transformed_scene['scene_dict']['agent_masks'].to(device="cuda")
        
        #Forward and return results
        self.model.eval()
        pred_scores, mu, sigma = self.model.forward(X, context=context, adjacency=adjacency, mask=None,)
        pred_scores, mu, sigma = pred_scores.detach(), mu.detach(), sigma.detach()
        return transformed_scene, (pred_scores, mu, sigma)
    
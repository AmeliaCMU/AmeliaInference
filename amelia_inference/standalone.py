import os
import pickle
import json
import torch
import numpy as np
import pickle as pkl
from typing import Tuple, Dict
from easydict import EasyDict
from src.data.components.amelia_dataset import AmeliaDataset
# from src.models.components.amelia import AmeliaTF  # Context aware model
# from src.models.components.amelia_traj import AmeliaTraj  # Non context aware model
import src.utils.global_masks as G
# from geographiclib.geodesic import Geodesic
# from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting


class SocialTrajPred():
    def __init__(self, airport: str, model, dataloader: AmeliaDataset,
                 use_map: bool = True, device: torch.device = torch.device('cuda')):

        # Create output directory
        self.airport = airport

        # Load model and dataloaders
        self.model = model
        self.dataloader = dataloader
        self.load_assets()

        # Configure CUDA
        torch.set_printoptions(precision=10, threshold=None, edgeitems=None,
                               linewidth=None, profile=None, sci_mode=False)
        self.device = device
        # self.model.to(self.device)

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

        graph_file = os.path.join(self.dataloader.context_dir, self.airport, 'semantic_graph.pkl')
        with open(graph_file, 'rb') as f:
            temp_dict = pickle.load(f)
            self.dataloader.semantic_pkl[self.airport] = temp_dict
            self.dataloader.semantic_maps[self.airport] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
            self.dataloader.hold_lines[self.airport] = temp_dict['hold_lines']

        limits_file = os.path.join(self.dataloader.assets_dir, self.airport, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.dataloader.ref_data[self.airport] = EasyDict(json.load(fp))

        self.dataloader.limits[self.airport] = (
            self.dataloader.ref_data[self.airport].espg_4326.north,
            self.dataloader.ref_data[self.airport].espg_4326.east,
            self.dataloader.ref_data[self.airport].espg_4326.south,
            self.dataloader.ref_data[self.airport].espg_4326.west
        )

    def dict_to_tensor(self, dict):
        tensor_dict = {}
        for key, value in dict.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value).to(self.device)
        return tensor_dict

    def load_ckpt(self, ckpt_path: str, from_pickle: bool = False):
        """
        Converts pytorch lightning state dict to torch state dict by removing net.
        prefix and load this to the GPT module.
        """
        if (from_pickle):
            with open(ckpt_path, 'rb') as file:
                state_dict = pkl.load(file)
        else:
            checkpoint = torch.load(ckpt_path,  map_location=self.device, weights_only=False)
            state_dict = checkpoint['state_dict']
            state_dict = {k.partition('net.')[2]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def forward(self, scene_data, benchmark: bool = True, random_ego=False) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # NOTE: quick workaround. Need to fix later
        # Transform scene in local frame
        if 'benchmark' not in scene_data or scene_data['benchmark'] is None:
            transformed_scene = self.dataloader.transform_scene_data(scene_data, random_ego=random_ego)
            transformed_scene = self.dataloader.collate_batch([transformed_scene])
        else:
            benchmark = scene_data['benchmark']
            agent_ids = np.asarray(scene_data['agent_ids'])
            bench_agents = [bid for bid in benchmark['bench_agents'] if bid in agent_ids]
            agents_in_scene = np.asarray([np.where(agent_ids == bid)[0][0] for bid in bench_agents])
            if len(agents_in_scene) <= 1:
                return None, None
            transformed_scene = []
            for i in range(len(agents_in_scene)):
                tf_scene = self.dataloader.transform_scene_data_bench(
                    scene_data, agents_in_scene, ego_agent=i)
                transformed_scene.append(tf_scene)
            transformed_scene = self.dataloader.collate_batch(transformed_scene)

        # Prepare inputs
        Y = transformed_scene['scene_dict']['rel_sequences'].to(device=self.device)
        X = torch.zeros_like(Y).type(torch.float).to(device=self.device)
        X[:, :, :self.dataloader.hist_len] = Y[:, :, :self.dataloader.hist_len]
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]]
        context = transformed_scene['scene_dict']['context'].to(device=self.device)
        adjacency = transformed_scene['scene_dict']['adjacency'].to(device=self.device)
        ego_agent = transformed_scene['scene_dict']['ego_agent_id']
        masks = transformed_scene['scene_dict']['agent_masks'].to(device=self.device)

        # Forward and return results
        self.model.to(self.device)
        self.model.eval()
        pred_scores, mu, sigma = self.model.forward(X, context=context, adjacency=adjacency)
        pred_scores, mu, sigma = pred_scores.detach(), mu.detach(), sigma.detach()
        return transformed_scene, (pred_scores, mu, sigma)

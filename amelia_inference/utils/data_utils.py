import os
import torch
import pickle
import numpy as np
from tqdm import tqdm


def get_scene_list(file, airport_code, traj_dir, max_scenes=None, sorted=True):
    scene_dir = os.path.join(traj_dir, airport_code, file)
    scene_file_list = [f for f in os.listdir(scene_dir)]
    if sorted:
        scene_file_list = order_file_tags(scene_file_list)

    scenes = []
    if max_scenes is None or len(scene_file_list) < max_scenes:
        max_scenes = len(scene_file_list)

    for i in tqdm(range(0, max_scenes)):
        scene = os.path.join(scene_dir, scene_file_list[i])

        with open(scene, 'rb') as f:
            scene_info = pickle.load(f)
        scenes.append(scene_info)

    return scene_file_list, scenes


def order_file_tags(file_list):
    # Sort the file list based on the numeric part of the filenames
    return sorted(file_list, key=lambda x: int(x.split('.')[0]))


def pad_array(array, padding, dim=1, device="cuda"):
    shape = list(array.shape)
    shape[dim] = padding
    shape = tuple(shape)
    try:
        bottom_padding = torch.zeros(shape).to(device=device)
    except:
        raise Exception
    padded_array = torch.cat((array, bottom_padding), dim=1)
    return padded_array


def get_full_scene_batch(batch, scene_dict, predictions, device):
    sequences = scene_dict['agent_sequences']
    agent_types = np.array(scene_dict['agent_types'])

    agents_in_scene = batch['scene_dict']['agents_in_scene'].detach().cpu().numpy().astype(int).tolist()
    other_agents = [i for i in range(scene_dict['num_agents']) if i not in agents_in_scene]

    # Separate agents in scene from unused agents
    other_agents_seq = sequences[other_agents]
    k_agents = sequences[agents_in_scene]

    other_agent_types = agent_types[other_agents].tolist()
    k_agent_types = agent_types[agents_in_scene].tolist()

    types = k_agent_types + other_agent_types
    all_agents = np.concatenate((k_agents, other_agents_seq), axis=0)
    N, D, T = all_agents.shape
    all_agents = all_agents.reshape(1, N, D, T)
    # Process ground truth
    (pred_scores, mu, sigma) = predictions
    B, K, d = pred_scores.shape

    if N > K:
        pred_scores = pad_array(pred_scores, padding=N-K, dim=1, device=device)
        mu = pad_array(mu, padding=N - K, dim=1, device=device)
        sigma = pad_array(sigma, padding=N - K, dim=1, device=device)

    agent_indexes = np.arange(0, K)
    batch['scene_dict']['sequences'] = torch.from_numpy(all_agents)
    batch['scene_dict']['agent_types'] = torch.tensor(types)
    batch['scene_dict']['num_agents'] = np.array([N])
    batch['scene_dict']['agents_in_scene'] = torch.from_numpy(agent_indexes)
    # Pad predictions
    return batch, (pred_scores, mu, sigma)

import os
import torch
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from standalone import SocialTrajPred
from src.utils.utils import plot_scene_batch
from src.data.components.amelia_dataset import AmeliaDataset
from src.models.components.amelia import AmeliaTF # Context aware model
from src.models.components.amelia_traj import AmeliaTraj # Non context aware model
from geographiclib.geodesic import Geodesic


from utils.config_utils import get_config_dict, get_model_config, get_data_config, setup_logger
from utils.data_utils import get_scene_list, get_full_scene_batch
from utils.common import *

# Load model and checkpoint
def get_inference_time(sequences, model, test_config, frame_limits):
    start_frame, end_frame = frame_limits
    print('---- Starting GPU Warmup ----')
    for i in range(0,10):
        test_seq = sequences[i]
        if test_seq.size != 0: pred_scores, mu, sigma = model.forward(test_seq,False,f'test_{i}')
    timings= np.zeros(((end_frame - start_frame),1))
    
    # Forward the remaining frames
    for i in tqdm(range(start_frame , end_frame)):
        test_seq = sequences[i]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if test_seq.size != 0: pred_scores, mu, sigma = model.forward(test_seq, test_config['plot'],
                                                                        f'test_{i}', test_config['save_scene'])
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        timings[i] = curr_time
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print('Mean: ',mean_syn)
    print('Std: ',std_syn)
    
def set_seeds(seed = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='default')
    parser.add_argument('--dataloader', type=str, default='default')
    args = parser.parse_args()
    
    set_seeds()
    test_config = get_config_dict(f"./configs/tests/{args.test}.yaml")
    data_config = get_data_config(f"./configs/data/{args.dataloader}.yaml", test_config)
    model_config = get_model_config(test_config['model_config_path'], data_config)
    dataloader = AmeliaDataset(data_config.config)
    if(test_config['use_map']):
         model = AmeliaTF(model_config)
    else:
         model = AmeliaTF(model_config)

    # Load extra configuration parameters
    tag = test_config['visualization_tag']
    from_pickle = test_config['from_pickle']
    plot = test_config['plot']
    tag = test_config['visualization_tag']
    
    # Load models
    predictor = SocialTrajPred(test_config['airport'], model= model, dataloader= dataloader,
                                use_map = test_config['use_map'])
    predictor.load_ckpt(test_config['weights_path'], from_pickle)
    
    testset = test_config['scene_file']
    for test_file in testset:
        print(f"----- Loading scenes for {test_file} -----")
        vis_tag = f"{test_file}_{tag}"
        out_dir = os.path.join('./out', f"{test_file}_{tag}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        scene_file_list, scenes = get_scene_list(test_file, test_config['airport'], 
                                                 traj_dir= TRAJ_DATA_DIR, 
                                                 max_scenes=test_config['max_scenes'],
                                                 sorted= False)
        
        print(f"----- Forwarding scenes -----")
        for scene in tqdm(scenes):
            batch, predictions = predictor.forward(scene)  
            if plot:
                full_scene, preds = get_full_scene_batch(batch, scene, predictions)
                plot_scene_batch(
                    asset_dir = data_config.config.assets_dir,
                    batch = full_scene,
                    predictions = preds,
                    hist_len = model.hist_len,
                    geodesic = Geodesic.WGS84,
                    tag = vis_tag,
                    plot_full_scene= True, 
                    out_dir = out_dir
                )
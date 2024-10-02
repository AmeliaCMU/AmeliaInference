import os
import torch
import random
import numpy as np
from tqdm import tqdm
import cv2
import imageio.v2 as imageio
import json

import hydra
from omegaconf import DictConfig


from amelia_inference.standalone import SocialTrajPred

try:
    from src.data.components.amelia_dataset import AmeliaDataset
    from src.models.components.amelia import AmeliaTF  # Context aware model
    from src.models.components.amelia_traj import AmeliaTraj  # Non context aware model
except ImportError:
    from amelia_tf.data.components.amelia_dataset import AmeliaDataset
    from amelia_tf.models.components.amelia import AmeliaTF  # Context aware model
    from amelia_tf.models.components.amelia_traj import AmeliaTraj  # Non context aware model
    import amelia_tf.utils.utils as U


from amelia_scenes.visualization import scene_viz
from amelia_scenes.utils.dataset import load_assets

from amelia_inference.utils.data_utils import get_scene_list


# Load model and checkpoint


def get_inference_time(sequences, model, test_config, frame_limits):
    start_frame, end_frame = frame_limits
    print('---- Starting GPU Warmup ----')
    for i in range(0, 10):
        test_seq = sequences[i]
        if test_seq.size != 0:
            pred_scores, mu, sigma = model.forward(test_seq, False, f'test_{i}')
    timings = np.zeros(((end_frame - start_frame), 1))

    # Forward the remaining frames
    for i in tqdm(range(start_frame, end_frame)):
        test_seq = sequences[i]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if test_seq.size != 0:
            pred_scores, mu, sigma = model.forward(test_seq, test_config['plot'],
                                                   f'test_{i}', test_config['save_scene'])
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        timings[i] = curr_time
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print('Mean: ', mean_syn)
    print('Std: ', std_syn)


def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(version_base="1.2", config_path="configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:

    dataloader = AmeliaDataset(cfg.data.config)
    set_seeds(cfg.seed)

    if cfg.tests.use_map:
        model = AmeliaTF(cfg.model.config)
    else:
        model = AmeliaTraj(cfg.model.config)

    # Load extra configuration parameters
    from_pickle = cfg.tests.from_pickle
    plot = cfg.tests.plot
    scene_type = cfg.tests.scene_type

    max_scenes = cfg.tests.max_scenes

    device = torch.device(cfg.device.accelerator)
    # Load models
    predictor = SocialTrajPred(
        cfg.tests.airport, model=model, dataloader=dataloader, use_map=cfg.tests.use_map, device=device)
    predictor.load_ckpt(cfg.tests.ckpt_path, from_pickle)
    # Load assets
    assets = load_assets(cfg.paths.base_dir, cfg.tests.airport)

    print(f"----- Loading scenes for {cfg.tests.scene_file} -----")
    file_tag = f"{cfg.tests.scene_file}_{cfg.tests.tag}"
    out_dir = os.path.join(cfg.tests.out_data_dir, f"{cfg.tests.scene_file}_{cfg.tests.tag}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    scene_file_list, scenes = get_scene_list(
        cfg.tests.scene_file, cfg.tests.airport, traj_dir=cfg.paths.proc_scenes,
        max_scenes=max_scenes, sorted=False)

    hist_len = cfg.data.config.hist_len

    plot_all = cfg.tests.plot_all

    print("----- Forwarding scenes -----")

    for scene in tqdm(scenes):
        if cfg.tests.scene_type == 'benchmark':
            scene['benchmark'] = {
                'bench_agents': cfg.tests.benchmark.agents
            }
        batch, predictions = predictor.forward(scene, collision=True)
        if batch is None:
            continue
        if plot:
            scenario_id = scene['scenario_id']
            filename = os.path.join(out_dir, f"{cfg.tests.airport}_scene-{scenario_id}_{file_tag}.png")
            scene['hist_len'] = hist_len
            scene['ego_agent_ids'] = batch['scene_dict']['ego_agent_id']
            scene['agents_in_scene'] = batch['scene_dict']['agents_in_scene']
            scores, mus, sigmas = predictions
            predictions = scores, mus, sigmas, batch['scene_dict']['sequences']
            scene_viz.plot_scene(
                scene, assets, filename, scene_type=scene_type, predictions=predictions, dpi=400, plot_all=plot_all, coll_threshold=cfg.tests.collision.threshold)


if __name__ == '__main__':
    main()

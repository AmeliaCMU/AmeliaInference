import cv2 
import json
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
from src.utils.data_utils import process_data_from_csv_ma, dotdict
from src.utils.visualization import plot_simple_movement

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--airport', type=str, default='default')
    parser.add_argument('--data_dir', type=str, default='default')
    parser.add_argument('--out_dir', type=str, default='default')
    args = parser.parse_args()
    hll_idx  = [False, False, True, True, True, False, False, False, False]

    map_filepath = f"assets/plotting/{args.airport}"
    limits_filepath = f"assets/plotting/{args.airport}/limits.json"
    # Read semantic map and reference info
    with open(limits_filepath, 'r') as fp:
        reference_data = dotdict(json.load(fp))
    theme = 'basic'
    ll_extent = (reference_data.north, reference_data.east, reference_data.south, reference_data.west)
    # Read background for visualization
    bkg_ground = cv2.imread(map_filepath + f'/bkg_map.png')
    bkg_ground = cv2.cvtColor(bkg_ground, cv2.COLOR_BGR2RGB)
    bkg_ground = cv2.resize(bkg_ground, (bkg_ground.shape[0]//2, bkg_ground.shape[1]//2))
    ac_map =  imageio.imread( f'assets/plotting/plane_icon.png')
    alt_plane = imageio.imread( f'assets/plotting/alt_plane_icon.png')
    maps = (bkg_ground, ac_map, alt_plane)
    
    sequences, id_in_scene = process_data_from_csv_ma(args.data_dir, 10)
    start_frame = 100
    end_frame = len(sequences)
    for i in tqdm(range(start_frame , end_frame, 1)):
        test_seq = sequences[i]
        if test_seq.size != 0: 
            test_seq = np.array(test_seq)
            plot_simple_movement(test_seq[:, :, hll_idx], maps, ll_extent,f'labeled_{i}', args.out_dir, ids_in_scene= id_in_scene[i], interest_ids=[1830679, 1830659])
            
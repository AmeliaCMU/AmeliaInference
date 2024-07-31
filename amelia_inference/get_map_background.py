import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from contextily import Place
import imageio.v2 as imageio
import rasterio
from rasterio.plot import show as rioshow

# plt.rcParams["figure.dpi"] = 800

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_osm_background(limits):
    north, east, south, west = limits
    zoom = 18
    print(ctx.howmany(west, south, east, north, zoom , ll=True))
    style = ' positron_extended_limits'
    airport_img, airport_ext = ctx.bounds2raster(west,
                                     south,
                                     east,
                                     north,
                                     f'ksea_{style}.tif',
                                     zoom=zoom,
                                     ll=True,
                                     max_retries = 6,
                                     wait = 0,
                                     source=ctx.providers.CartoDB.Positron)
    
    # airport_img, airport_ext = ctx.warp_tiles(airport_img, (west, south, east, north), t_crs='EPSG:4326')
    # im = Image.fromarray(airport_img, mode='F') # float32
    # airport_img.save("test2.tiff", "TIFF")
    
if __name__ == '__main__':
    airport = 'ksea'
    map_filepath = f"maps/{airport}"
    limits_filepath = f"maps/{airport}/limits.json"
    # Read semantic map and reference info
    with open(limits_filepath, 'r') as fp:
        reference_data = dotdict(json.load(fp))
        offset = 0.004
        reference_data.extended_north = reference_data.north
        reference_data.extended_east  = reference_data.east + offset
        reference_data.extended_south = reference_data.south - offset
        reference_data.extended_west  = reference_data.west - offset
        
    with open(limits_filepath, 'w') as fp:
        fp.write(json.dumps(reference_data))
        
    ll_limits = (reference_data.north, reference_data.east + offset , reference_data.south - offset, reference_data.west - offset)
        
    get_osm_background(ll_limits)
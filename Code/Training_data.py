import tensorflow as tf
import rasterio
import numpy as np
import os
from tqdm import tqdm

# AnnualCrop
Path_c="./data/EuroSAT_MS"
label=[]
rasters=[]

for category,index_label in tqdm(zip(sorted(os.listdir(Path_c)),range(len(os.listdir(Path_c)))),desc='Main Process',position=0):
    for images in tqdm(os.listdir(Path_c+"/"+category),leave=False, position=1,desc=category):
        label.append(index_label)
        with rasterio.open(Path_c+"/"+category+"/"+images) as src:
            raster=src.read()
            raster=np.moveaxis(raster,0,-1)
        rasters.append(raster)

rasters=np.stack(rasters)

np.savez("labeled_images.npz",x=rasters,y=label)


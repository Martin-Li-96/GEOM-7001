import keras
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio.crs import CRS

model=keras.saving.load_model("./RGB_CNN/RGB_CNN-96-0.85.keras")

#year
year="2016"

B02=rasterio.open("./Targets_int16/{year}-01/{year}-01-B02.tif".format(year=year)).read()
B03=rasterio.open("./Targets_int16/{year}-01/{year}-01-B03.tif".format(year=year)).read()
B04=rasterio.open("./Targets_int16/{year}-01/{year}-01-B04.tif".format(year=year)).read()


Target=np.stack([B04,B03,B02],axis=0)
Target=np.moveaxis(Target,0,-1)

# Target=(Target+ 32768).astype(np.uint16)
# Target=Target.astype(np.uint16)
Target=np.clip(Target, 0, 65535).astype(np.uint16)

#
# min_int16 = np.iinfo(np.int16).min  # -32768
# max_int16 = np.iinfo(np.int16).max  # 32767
# max_uint16 = np.iinfo(np.uint16).max  # 65535
#
# # Step 1: Normalize the int16 values to the range [0, 1]
# image_normalized = (Target - min_int16) / (max_int16 - min_int16)
#
# # Step 2: Rescale the normalized values to uint16 range [0, 65535]
# Target = (image_normalized * max_uint16).astype(np.uint16)

#Padding

# (7075,7824)

Result=np.empty((7104,7856),np.uint16)


Target=np.pad(Target, ((0,0),(0, 29), (0,32), (0, 0)), mode='constant')
window_size=64
step_size=16

t7=[]

for row in tqdm(range((Target.shape[1] - window_size) // step_size + 1)):
    for col in range((Target.shape[2] - window_size) // step_size + 1):
        # Extract the window from the input Target array
        window = Target[:, row * step_size:row * step_size + window_size, col * step_size:col * step_size + window_size,
                 :]
        prediction = np.argmax(model.predict(window,verbose=0), axis=-1)

        #Overlap pixel by new value
        Result[row * step_size:row * step_size + window_size,
        col * step_size:col * step_size + window_size] = prediction

        # Increment the weight array (each pixel gets a weight of 1 from each window it appears in)

Result=Result[:7075,:7824]



crs=rasterio.open("./Targets_int16/{year}-01/{year}-01-B01.tif".format(year=year)).crs
transform=rasterio.open("./Targets_int16/{year}-01/{year}-01-B01.tif".format(year=year)).transform
with rasterio.open(
    './classified_target/{year}_RGB_CNN_include7.tif'.format(year=year),  # Output file name
    'w',  # Write mode
    driver='GTiff',  # GeoTIFF format
    height=Result.shape[0],  # Number of rows (height)
    width=Result.shape[1],  # Number of columns (width)
    count=1,  # Number of bands (for multiband arrays, this will be > 1)
    dtype=Result.dtype,  # Data type (must match the numpy array dtype)
    crs=crs,  # Coordinate reference system (UTM zone 56S)
    transform=transform,  # Georeferencing transform
) as dst:
    dst.write(Result, 1)


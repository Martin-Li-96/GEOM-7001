import keras
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio.crs import CRS

model=keras.saving.load_model("./VGG19/fold1-130-0.9044.keras")

#year
year="2016"

B01=rasterio.open("./Targets_int16/{year}-01/{year}-01-B01.tif".format(year=year)).read()
B02=rasterio.open("./Targets_int16/{year}-01/{year}-01-B02.tif".format(year=year)).read()
B03=rasterio.open("./Targets_int16/{year}-01/{year}-01-B03.tif".format(year=year)).read()
B04=rasterio.open("./Targets_int16/{year}-01/{year}-01-B04.tif".format(year=year)).read()
B05=rasterio.open("./Targets_int16/{year}-01/{year}-01-B05.tif".format(year=year)).read()
B06=rasterio.open("./Targets_int16/{year}-01/{year}-01-B06.tif".format(year=year)).read()
B07=rasterio.open("./Targets_int16/{year}-01/{year}-01-B07.tif".format(year=year)).read()
B08=rasterio.open("./Targets_int16/{year}-01/{year}-01-B08.tif".format(year=year)).read()
B8A=rasterio.open("./Targets_int16/{year}-01/{year}-01-B8A.tif".format(year=year)).read()
B09=rasterio.open("./Targets_int16/{year}-01/{year}-01-B09.tif".format(year=year)).read()
B10=rasterio.open("./Targets_int16/{year}-01/{year}-01-B10.tif".format(year=year)).read()
B11=rasterio.open("./Targets_int16/{year}-01/{year}-01-B11.tif".format(year=year)).read()
B12=rasterio.open("./Targets_int16/{year}-01/{year}-01-B12.tif".format(year=year)).read()


Target=np.stack([B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12],axis=0)
Target=np.moveaxis(Target,0,-1)

Target=np.clip(Target, 0, 65535).astype(np.uint16)



Result=np.empty((7104,7856),np.uint16)


Target=np.pad(Target, ((0,0),(0, 29), (0,32), (0, 0)), mode='constant')
window_size=64
step_size=16

t7=[]

for row in tqdm(range((Target.shape[1] - window_size) // step_size + 1)):
    for col in range((Target.shape[2] - window_size) // step_size + 1):
        window = Target[:, row * step_size:row * step_size + window_size, col * step_size:col * step_size + window_size,
                 :]

        # Predict the class or probability for the window
        prediction = np.argmax(model.predict(window,verbose=0), axis=-1)
        if prediction == 7:
            prediction=4


        Result[row * step_size:row * step_size + window_size,
        col * step_size:col * step_size + window_size] = prediction


Result=Result[:7075,:7824]



crs=rasterio.open("./Targets_int16/{year}-01/{year}-01-B01.tif".format(year=year)).crs
transform=rasterio.open("./Targets_int16/{year}-01/{year}-01-B01.tif".format(year=year)).transform
with rasterio.open(
    './classified_target/{year}_VGG19.tif'.format(year=year),  # Output file name
    'w',
    driver='GTiff',
    height=Result.shape[0],
    width=Result.shape[1],
    count=1,
    dtype=Result.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(Result, 1)


# import keras
# import numpy as np
# import rasterio
import openeo
# import json
from tqdm import tqdm
import geopandas as gpd

#********************************
#2024-08-26 v1
#2024-08-31
#2016-08-28
#*******************************************

connection = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2")
connection.authenticate_oidc()



gdf=gpd.read_file("./data/Ward_boundaries.shp")
bounding_box=gdf.to_crs(epsg=4326).total_bounds


aoi_bbox = {
    "west": bounding_box[0],
    "south": bounding_box[1],
    "east": bounding_box[2],
    "north": bounding_box[3]
}


for band in tqdm([
         "B02", "B03", "B04",
    ]):


    process = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=aoi_bbox,
        bands=band,
        temporal_extent=["2024-08-31", "2024-08-31"]
    )

    process.download("./Targets_int16/2024-01/2024-01-{b}.tif".format(b=band),format="GTIFF")

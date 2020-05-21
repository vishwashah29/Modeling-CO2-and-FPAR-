from netCDF4 import Dataset

nc = Dataset("C:/Users/VISHWA/Desktop/Guj1.nc4",'r')

nc.variables.keys()

lat = nc.variables['latitude'][:].data
print(lat[:8],type(lat[:8]))
lon = nc.variables['longitude'][:].data
print((((68 < lon) & (lon < 75)) & (20 <lat) & (lat<25)).sum())
print(lon[(68 < lon) & (lon < 75)&(20<lat) & (lat<25)])

co2 = nc.variables['xco2'][:].data
print(co2)


import pandas as pd
import numpy as np
from pykrige import OrdinaryKriging

model = OrdinaryKriging(lon, lat, co2,variogram_model='gaussian')
#print(model)

# grid around gujarat
gridx = np.arange(68.1862489922972372, 74.4766299192354495, 0.1)
print(gridx.ndim)
print(gridx.shape)
print(len(gridx))
gridy = np.arange(20.1209430201043347, 24.7084824408120198, 0.1)
print(gridy.shape)
coords = np.transpose([np.tile(gridx, len(gridy)), np.repeat(gridy, len(gridx))]) #for every combination of lat and long together.
#array([[1, 5],
#       [2, 5],
#       [3, 5],
#       [1, 6],
#       [2, 6],
#       [3, 6],
#       [1, 7],
#       [2, 7],
#       [3, 7],
#       [1, 8],   b=np.array([5,6,7,8]) a=np.array([1,2,3])
#       [2, 8],
#       [3, 8]]) array c=c=np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))]) 
#
z, ss = model.execute('grid', gridx, gridy)
print(z)
print("hey")
print(ss)
#
#write gujarat co2 data to csv
pd.DataFrame(np.c_[coords[:, 0], coords[:, 1], z.ravel()]).to_csv("guj2_co2.csv")


# Write gujarat co2 data to raster file 
import rasterio

guj_ras = rasterio.open('C:/Users/VISHWA/Desktop/guj_raster.tif')

co2_guj = rasterio.open("co2_guj.tif", 'w',driver="GTiff", width=guj_ras.width, height=guj_ras.height, count=1, dtype=z.dtype, crs=guj_ras.crs, transform=guj_ras.transform)

co2_guj.write(z.reshape((1, z.shape[0], z.shape[1])))
co2_guj.close()

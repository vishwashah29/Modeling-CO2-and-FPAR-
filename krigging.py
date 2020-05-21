import numpy as np
from netCDF4 import Dataset
import cv2
from pykrige import OrdinaryKriging
import rasterio
import pandas as pd
#import statistics
import matplotlib.pyplot as plt

#read data from nc file for co2
nc=Dataset("C:/Users/VISHWA/Desktop/EVS_model/AIRS.2010.01.01.hdf.nc4",'r')
print(nc.variables.keys())

lat = nc.variables['Latitude'][:].data

#print(lat[:8],type(lat[:8]))
lon = nc.variables['Longitude'][:].data
lon=np.tile(lon,len(lat))
lat=np.repeat(lat,3)
print(lon)
print(lat)

#print((((68 < lon) & (lon <= 75)) & (20 <lat) & (lat<=26)).sum())
#print(lat[(68 < lon) & (lon < 75)&(20<lat) & (lat<=26)])

co2 = nc.variables['mole_fraction_of_carbon_dioxide_in_free_troposphere'][:].data
co2=(10**6)*co2
co2=co2.flatten()
#co2=co2.transpose()
print(co2)

gridx = np.linspace(70.0, 75.0, 737)
#print(gridx)
gridy = np.linspace(22.0, 26.0, 448)
#print(gridy)

coords = np.transpose([np.tile(gridx, len(gridy)), np.repeat(gridy, len(gridx))])
model = OrdinaryKriging(lon, lat, co2,variogram_model='linear',enable_plotting=True)
z,ss=model.execute('grid',gridx,gridy)
#
##read data from tif images for fpar values
fpar= rasterio.open('2010_01_01.tif')
#print(fpar.width)
fpar=fpar.read(1)
fpar=fpar.flatten()
print(len(fpar))

#
ndvi=rasterio.open('2010_01_01-737x448-ndvi.tif')
ndvi=ndvi.read(1)
ndvi=ndvi.flatten()
print(len(ndvi))

gpp=rasterio.open('2010_01_01-gpp.tif')
gpp=gpp.read(1)
gpp=gpp.flatten()
print(len(gpp))

#removing Nan values from the data

df=pd.DataFrame(np.c_[coords[:, 0], coords[:, 1], z.ravel(),fpar,ndvi,gpp])
df.columns=['longitude','latitude','co2','fpar','ndvi','gpp']
df=df.apply(pd.to_numeric, errors='coerce')
df=df.dropna()
df = df.reset_index(drop=True)
df.to_csv("final_values_ndvi_gpp_linear.csv")

#pd.DataFrame(np.c_[coords[:, 0], coords[:, 1], z.ravel(),band1]).to_csv("gujkrig3_co2.csv")









#i = 0
#j = 0
#
#fin_arr = np.zeros(shape = (9,3), dtype=float)
#cnt = 0
#print("Hey",fin_arr.shape)
#while i < len(xco2):
#    j = 0
#    while j < len(xco2[i,:]):
#        if (xco2[i][j] < 1000):
#            print(xco2[i][j])
#        else:
#            xco2[i][j] = temp_avg
#        arr = np.array([lon[j], lat[i], xco2[i][j]])
#        fin_arr[cnt, :] = fin_arr[cnt, :] + arr
#        cnt = cnt + 1
#        j = j + 1
#    i = i + 1
#    
#print(fin_arr)

# making the grid according to the geotiff image of fpar so we can fit the co2 values in its dimensions
#


#model = OrdinaryKriging(lon, lat, co2,variogram_model='gaussian')
# implementing kriging onto the co2 values

#OK = OrdinaryKriging(fin_arr[:, 0], fin_arr[:, 1], fin_arr[:, 2], variogram_model='gaussian', enable_plotting=True)
#
#z, ss = OK.execute('grid', gridx, gridy)





#model = OrdinaryKriging(lon, lat, co2,variogram_model='gaussian')
#
#coords = np.transpose([np.tile(gridx, len(gridy)), np.repeat(gridy, len(gridx))]) #for every combination of lat and long together.
#z, ss = model.execute('grid', gridx, gridy)
#
#print(z)

#pd.DataFrame(np.c_[gridx[:, 0], [:, 1], z.ravel()]).to_csv("guj2_co2.csv")

#fparavg=band1.mean() can't do mean because nan values
#print(fparavg)
#print(band1.shape) #(330176= 448*737)

#i=1
#for i in range(len(band1)):
#    if(band1[i]==NaN):
    
    

#removing Nan values from the data
#df=pd.DataFrame(band1)
#df=df.apply(pd.to_numeric, errors='coerce')
#df=df.dropna()
#df = df.reset_index(drop=True)
#df=df.flatten()
#print (df.dtype)
#fpar_avg=math.m

#useless stuff
#img_path="2010_01_01.tif"
#image=cv2.imread(img_path)
#print(len(image))
#cv2.imshow("guj",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
##

#now writing a krigging model to get co2 data match with dimension of fpar


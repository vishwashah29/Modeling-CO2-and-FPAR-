import netCDF4 
import numpy as np
import pandas as pd

nc=netCDF4.Dataset('C:/Users/VISHWA/Desktop/Guj1.nc4','r')
#print(data)

#print(data.variables.keys())

#co2=data.variables['xco2'][:]
#print(co2)



#//precip_nc_file = 'file path'
#//nc = netCDF4.Dataset(precip_nc_file, mode='r')

print(nc.variables.keys())
lat = nc.variables['latitude'][:]
#print(lat)
lon = nc.variables['longitude'][:]
#print(lon)
time_var = nc.variables['time']
#print(time_var)
dtime = netCDF4.num2date(time_var[:],time_var.units)
#print(dtime)
precip = nc.variables['xco2'][:]
print(precip)
#data1=pd.read(precip)
#colunslist=data1.head()
#colunslist
#precip_ts = pd.DataFrame(precip, columns=['co2'])
#print(precip_ts)
#precip_ts['longitude'] = lon
#precip_ts['latitude'] = lat
#precip_ts['time'] = dtime
#precip_ts.to_csv('co2guj.csv',index=True, header=True)
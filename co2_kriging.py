import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import netCDF4


# adding the path of the file
file_path = 'C:/Users/VISHWA/Desktop/EVS_model/AIRS.2010.01.01.hdf.nc4'

nc = netCDF4.Dataset(file_path, mode='r')

print(nc.variables.keys())

# getting the variables into a numpy array
xco2 = nc.variables['mole_fraction_of_carbon_dioxide_in_free_troposphere'][:]
xco2 = xco2 * 1000000
long = nc.variables['Longitude'][:]
lat = nc.variables['Latitude'][:]

print(long)
print('\n')
print(lat)
print('\n')
print(xco2.shape)
print(len(xco2)) #prints the number of rows.


# creating fin_arr which is a 2D array which will be passed in the function calling of pykrige

i = 0
j = 0

fin_arr = np.zeros(shape = (9,3), dtype=float)
cnt = 0
print("Hey",fin_arr.shape)
while i < len(xco2):
    j = 0
    while j < len(xco2[i,:]):
        if (xco2[i][j] < 1000):
            print(xco2[i][j])
        else:
            xco2[i][j] = temp_avg
        arr = np.array([long[j], lat[i], xco2[i][j]])
        fin_arr[cnt, :] = fin_arr[cnt, :] + arr
        cnt = cnt + 1
        j = j + 1
    i = i + 1
    
print(fin_arr)

# making the grid according to the geotiff image of fpar so we can fit the co2 values in its dimensions

gridx = np.linspace(70.0, 75.0, 737)
print(gridx)
gridy = np.linspace(22.0, 26.0, 448)
print(gridy)

# implementing kriging onto the co2 values

OK = OrdinaryKriging(fin_arr[:, 0], fin_arr[:, 1], fin_arr[:, 2], variogram_model='gaussian', enable_plotting=True)
z, ss = OK.execute('grid', gridx, gridy)

# z is the 2D array of co2 which matches with the same dated raster converted 2D array of fpar
# each cell of rasterArray.data[i][j] (correspoonds to)-> z[i][j]
print(z)

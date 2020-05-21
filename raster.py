import rasterio
import numpy as np
import cv2
import pandas as pd
import sklearn



dataset = rasterio.open('2018_02_02.tif')

#image=cv2.imread('2018_02_02.tif')
#cv2.imshow("image",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


band1=dataset.read(1)
print(band1)

df=pd.DataFrame(band1)
df=df.apply(pd.to_numeric, errors='coerce')
df=df.dropna()
df = df.reset_index(drop=True)

print (df)


#stats=[]


#stats.append({
#            'min':band1.min(),
#            'mean':band1.mean(),
#            'median':np.median(band1),
#            'max':band1.max()
#                    })
#
#print(stats)

#x = np.linspace(-4.0, 4.0, 240)
#y = np.linspace(-3.0, 3.0, 180)
#X, Y = np.meshgrid(x, y)
#Z1 = np.exp(-2 * np.log(2) * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 1 ** 2)
#Z2 = np.exp(-3 * np.log(2) * ((X + 0.5) ** 2 + (Y + 0.5) ** 2) / 2.5 ** 2)
#Z = 10.0 * (Z2 - Z1)
#
#with rasterio.open(
#    '2018_02_02.tif',
#    'w',
#    driver='GTiff',
#    height=Z.shape[0],
#    width=Z.shape[1],
#    count=1,
#    dtype=Z.dtype,
#    crs='+proj=latlong',
#    transform=transform,
#) as dst:
#    dst.write(Z, 1)

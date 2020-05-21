import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv(r"final_values_ndvi_gpp_linear.csv")
print(df.shape)

frac_data=df.sample(frac=0.003)

y1=frac_data['ndvi'].values.reshape(-1,1)
x1 = frac_data['fpar'].values.reshape(-1,1)
reg1 = LinearRegression()
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)
reg1.fit(x1_train, y1_train)

y1_pred = reg1.predict(x1_test)

plt.scatter(x1, y1, color='blue')
plt.plot(x1_test, y1_pred, color='red')
plt.xlabel('FPAR')
plt.ylabel('NDVI')
plt.show()

r2_score(y1_test, y1_pred)
print(r2_score)

y2 = frac_data['gpp'].values.reshape(-1,1)
x2 = frac_data['fpar'].values.reshape(-1,1)
reg2 = LinearRegression()
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)
reg2.fit(x2_train, y2_train)

y2_pred = reg2.predict(x2_test)

plt.scatter(x2, y2, color='blue')
plt.plot(x2_test, y2_pred, color='red')
plt.xlabel('FPAR')
plt.ylabel('GPP')
plt.show()

y3 = frac_data['gpp'].values.reshape(-1,1)
x3 = frac_data['ndvi'].values.reshape(-1,1)
reg3 = LinearRegression()
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2)
reg3.fit(x3_train, y3_train)

y3_pred = reg3.predict(x3_test)

plt.scatter(x3, y3, color='blue')
plt.plot(x3_test, y3_pred, color='red')
plt.xlabel('NDVI')
plt.ylabel('GPP')
plt.show()

r2_score(y3_test, y3_pred)
print(r2_score)


y = frac_data['co2']
x = frac_data[['fpar', 'ndvi', 'gpp']]

reg = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

reg.score(x_test, y_test)
r2_score(y_test,y_pred)
print("r2_score final",r2_score)

plt.scatter(x['fpar'], y, color='green')
#plt.plot(x_test['fpar'], y_pred, color='red')
plt.xlabel('FPAR')
plt.ylabel('CO2')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset=pd.read_csv("/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Random_Forest_Trial/Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(X,Y)

#prediction
y_pred=regressor.predict(X)

#visualisation of the random forest
plt.scatter(X,Y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title('Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#high resolution
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


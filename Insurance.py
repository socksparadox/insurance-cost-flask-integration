import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("insurance.csv")
data.head()
col = data.columns

data.sex = data.sex.astype('category').cat.codes
data.smoker = data.smoker.astype('category').cat.codes
print(data.region.value_counts())

data.region = [1 if x == 'southeast' else 2 if x =='southwest' 
               else 3 if x == 'northwest' else 4 for x in data.region]
data.info()
data.isnull().sum()

X = data.drop('charges', axis = 1)
y = data['charges']

def accuracy_score(model):
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    mse = (mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return [(train_acc * 100), (test_acc * 100) , rmse, r2]

X_train, X_test, y_train, y_test = train_test_split(X, y)

linr = LinearRegression()
linr.fit(X_train, y_train)
Linear_Regression = accuracy_score(linr)

dtr = DecisionTreeRegressor(max_depth = 3)
dtr.fit(X_train, y_train)
Decision_Tree_Regressor = accuracy_score(dtr)

rfr = RandomForestRegressor(max_depth = 4)
rfr.fit(X_train, y_train)
Random_Forest_Regressor = accuracy_score(rfr)

xgbr = XGBRegressor(max_depth = 2, learning_rate = 0.1)
xgbr.fit(X_train, y_train)
XGBoost_Regressor = accuracy_score(xgbr)

results = pd.DataFrame(columns = ['Training Accuracy', 'Testing Accuracy', 
                                  'Root Mean Squared Error', 'R2 Score'],
                      index = ['Linear Regression',  'Decision Tree Regression', 
                               'Random Forest Regression', 'XGBoost Regression'],
                      data = [Linear_Regression,Decision_Tree_Regressor,
                              Random_Forest_Regressor,XGBoost_Regressor])

print(results)

pickle.dump(rfr, open('model.pkl','wb'))

print(rfr.predict([[19,0,27.9,0,1,2]]))




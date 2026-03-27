from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import max_error, mean_squared_error, r2_score

data = pd.read_csv('data_C02_emission.csv')

categorical_variable = 'Fuel Type'
numeric_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)']
input_variables = [categorical_variable] + numeric_variables
output = 'CO2 Emissions (g/km)'

X = data[input_variables]
y = data[output]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=1)

ohe = OneHotEncoder()
X_categorical_train = ohe.fit_transform(X_train[[categorical_variable]]).toarray()
X_categorical_test = ohe.transform(X_test[[categorical_variable]]).toarray()

X_numeric_train = X_train[numeric_variables].to_numpy()
X_numeric_test = X_test[numeric_variables].to_numpy()

X_model_train = np.hstack((X_numeric_train, X_categorical_train))
X_model_test = np.hstack((X_numeric_test, X_categorical_test))

linearModel = lm.LinearRegression()
linearModel.fit(X_model_train, y_train)

y_test_p = linearModel.predict(X_model_test)

ME = max_error(y_test, y_test_p)
print(f"Max Error: {ME}")

rmse = np.sqrt(mean_squared_error(y_test, y_test_p))
r2 = r2_score(y_test, y_test_p)
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

absolute_errors = pd.Series(np.abs(y_test.values - y_test_p), index=y_test.index)
max_error_id = absolute_errors.idxmax()
max_error_value = absolute_errors.loc[max_error_id]

max_error_vehicle = data.loc[max_error_id, ['Make', 'Model']]
print(f"Max error: {max_error_value:.4f} g/km")
print(f"Vehicle with max error: {max_error_vehicle['Make']} {max_error_vehicle['Model']}")
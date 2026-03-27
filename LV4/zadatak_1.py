from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

# a)
data = pd.read_csv('data_C02_emission.csv')

input_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)']
output = 'CO2 Emissions (g/km)'

X = data[input_variables]
y = data[output]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=1)

# b)
fig, ax = plt.subplots(5, 2, figsize=(14, 18), sharey=True)

for i, feature in enumerate(input_variables):
    ax[i, 0].scatter(X_train[feature], y_train, c='Red')
    ax[i, 0].set_title(f'Train: {feature}')
    ax[i, 0].set_ylabel('CO2 Emissions (g/km)')

    ax[i, 1].scatter(X_test[feature], y_test, c='Blue')
    ax[i, 1].set_title(f'Test: {feature}')

ax[4, 0].set_xlabel('Input value')
ax[4, 1].set_xlabel('Input value')
fig.suptitle('b) CO2 emissions vs all input variables', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# c)
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)

fig, ax = plt.subplots(5, 2, figsize=(14, 18))

for i, feature in enumerate(input_variables):
    ax[i, 0].hist(X_train[feature], color='Red', alpha=0.7)
    ax[i, 0].set_title(f'Before scaling: {feature}')
    ax[i, 0].set_ylabel('Frequency')

    ax[i, 1].hist(X_train_n[:, i], color='Blue', alpha=0.7)
    ax[i, 1].set_title(f'After scaling: {feature}')

ax[4, 0].set_xlabel('Input value')
ax[4, 1].set_xlabel('Scaled value')
fig.suptitle('c) Input variable distributions before and after scaling', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

X_test_n = sc.transform(X_test)

# d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print(linearModel.coef_)

# e)
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)

line_min = min(y_test.min(), y_test_p.min())
line_max = max(y_test.max(), y_test_p.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--')

plt.title("Comparison of actual and predicted values")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

# f)
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = root_mean_squared_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}\n MSE: {MSE}\n MAPE: {MAPE}\n RMSE: {RMSE}\n R2: {R2}")

# g)

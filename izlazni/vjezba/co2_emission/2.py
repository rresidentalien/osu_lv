'''
- Zadaci uključuju:
  - Predobradu i analizu podataka
  - Implementaciju algoritama strojnog učenja
  - Provedbu učenja i testiranja modela
'''

import pandas as pd
data = pd.read_csv('data_C02_emission.csv')

# predobrada
data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

X = data.drop(columns='CO2 Emissions (g/km)')
y = data['CO2 Emissions (g/km)']

X = pd.get_dummies(X, columns=['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# linearna regresija - implementacija i učenje
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

# linearna regresija - testiranje
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
print('Train:')
y_pred = model.predict(X_train)
print('RMSE: ', root_mean_squared_error(y_train, y_pred))
print('MAE: ', mean_absolute_error(y_train, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_train, y_pred))
print('R2: ', r2_score(y_train, y_pred))

print('Test:')
y_pred = model.predict(X_test)
print('RMSE: ', root_mean_squared_error(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))


# knn - implementacija i učenje
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

# knn - testiranje
print('Train:')
y_pred = model.predict(X_train)
print('RMSE: ', root_mean_squared_error(y_train, y_pred))
print('MAE: ', mean_absolute_error(y_train, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_train, y_pred))
print('R2: ', r2_score(y_train, y_pred))

print('Test:')
y_pred = model.predict(X_test)
print('RMSE: ', root_mean_squared_error(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))

# svm - implementacija i učenje
from sklearn.svm import SVR
model = SVR().fit(X_train, y_train)

# svm - testiranje
print('Train:')
y_pred = model.predict(X_train)
print('RMSE: ', root_mean_squared_error(y_train, y_pred))
print('MAE: ', mean_absolute_error(y_train, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_train, y_pred))
print('R2: ', r2_score(y_train, y_pred))

print('Test:')
y_pred = model.predict(X_test)
print('RMSE: ', root_mean_squared_error(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))

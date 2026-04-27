from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)

# 0.0.2
# a)
print('Broj uzoraka: ', len(data.frame))
print('Broj značajki: ', len(data.frame.columns) - 1)
print('Značajke: ', data.frame.columns[:-1])
print('Izlazna veličina: ', data.frame.columns[-1])
print(f'Raspon vrijednosti izlazne veličine: od {data.frame['MedHouseVal'].min()} do {data.frame['MedHouseVal'].max()} stotina tisuća')

# b)
X = data.frame.drop(columns='MedHouseVal')
y = data.frame['MedHouseVal']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.corr())
print(data.frame.corr())

# c)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for ax, feature in zip(axes.flatten(), X.columns):
	ax.scatter(X[feature], y, color='green', alpha=0.5, s=10)
	ax.set_title(f'Ovisnost {feature} o MedHouseVal')
	ax.set_xlabel(feature)
	ax.set_ylabel('MedHouseVal')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for ax, feature in zip(axes.flatten(), X.columns):
	ax.boxplot(X[feature])
	ax.set_title(f'Distribucija: {feature}')
	ax.set_xlabel(feature)

plt.tight_layout()
plt.show()


# 0.0.3
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# a)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

# b)
y_test_pred = model.predict(X_test)

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
print('RMSE: ', root_mean_squared_error(y_test, y_test_pred))
print('MAE: ', mean_absolute_error(y_test, y_test_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_test_pred))
print('R2: ', r2_score(y_test, y_test_pred))

# c)
plt.scatter(y_test, y_test_pred, c='orange', s=10)
min_value = y_test.min()
max_value = y_test.max()
plt.plot([min_value, max_value], [min_value, max_value], color='gray')
plt.xlabel('Stvarni podaci')
plt.ylabel('Predviđeni podaci')
plt.title('Ovisnost stvarnih i predviđenih vrijednosti')
plt.show()


# 0.0.4
import keras
from keras import layers

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,))) # za 8 značajki koje imamo
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1))
model.summary()

# b)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# c)
callbacks = [
	keras.callbacks.TensorBoard(log_dir='logs/housing')
]

# d)
model.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=32, validation_split=0.1)

# e)
model.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=320, validation_split=0.1)

# g)
model.save('2.keras')

# h)
from keras.models import load_model

loaded_model = load_model('2.keras')
y_test_pred = loaded_model.predict(X_test)

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
print('RMSE: ', root_mean_squared_error(y_test, y_test_pred))
print('MAE: ', mean_absolute_error(y_test, y_test_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_test_pred))
print('R2: ', r2_score(y_test, y_test_pred))

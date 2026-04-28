from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import keras
from keras import layers
from keras.models import load_model

data = load_diabetes(as_frame=True)
X = data.data
y = data.target

# zadatak 1
print('Broj uzoraka: ', len(data.frame))
print('Broj značajki: ', len(data.feature_names))
print('Značajke: ', data.frame.columns[:-1])
print('Izlazna veličina: ', data.frame.columns[-1])
print(f'Raspon vrijednosti izlazne veličine: od {data.frame['target'].min()} do {data.frame['target'].max()}')

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
for ax, feature in zip(axes.flatten(), X.columns):
	ax.hist(X[feature])
	ax.set_xlabel(feature)
plt.suptitle('Histogram svih značajki')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
for ax, feature in zip(axes.flatten(), X.columns):
	ax.boxplot(X[feature])
	ax.set_xlabel(feature)
plt.suptitle('Kutijasti dijagram svih značajki')
plt.show()

fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', square=True, cbar=True)
ax.set_title('Korelacijska matrica značajki')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, feature in zip(axes.flatten(), data.feature_names):
	ax.scatter(X[feature], data.frame['target'], color='green', alpha=0.5, s=10)
	ax.set_title(f'Ovisnost {feature} o target')
	ax.set_xlabel(feature)
	ax.set_ylabel('target')
plt.tight_layout()
plt.suptitle('Ovisnost značajki o izlaznoj varijabli')
plt.show()

# zadatak 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# podaci su već skalirani pa ih ne moramo ponovno skalirati
model = LinearRegression().fit(X_train, y_train)

y_test_pred = model.predict(X_test)
print('RMSE: ', root_mean_squared_error(y_test, y_test_pred))
print('MAE: ', mean_absolute_error(y_test, y_test_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_test_pred))
print('R2: ', r2_score(y_test, y_test_pred))

plt.scatter(y_test, y_test_pred, c='orange', s=10)
min_value = y_test.min()
max_value = y_test.max()
plt.plot([min_value, max_value], [min_value, max_value], color='gray')
plt.xlabel('Stvarni podaci')
plt.ylabel('Predviđeni podaci')
plt.title('Ovisnost stvarnih i predviđenih vrijednosti')
plt.show()

# zadatak 3
model = keras.Sequential()
model.add(layers.Input(shape=(10,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

callbacks = [
	keras.callbacks.TensorBoard(log_dir='logs/diabetes')
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=32, validation_split=0.1)

model2 = model
model2.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=320, validation_split=0.1)

model.save('2.keras')

loaded_model = load_model('2.keras')
y_test_pred = loaded_model.predict(X_test)
print('RMSE: ', root_mean_squared_error(y_test, y_test_pred))
print('MAE: ', mean_absolute_error(y_test, y_test_pred))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_test_pred))
print('R2: ', r2_score(y_test, y_test_pred))

#imports
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import keras
from keras import layers
from keras.models import load_model
import numpy as np
from matplotlib.colors import ListedColormap

labels = {0: 'benign', 1: 'malign'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

#1. zadatak
data = load_breast_cancer(as_frame=True)
print('Broj uzoraka: ', len(data.frame))
print('Broj značajki: ', len(data.frame.columns) - 1)
print('Značajke: ', data.feature_names)
print('Izlazna veličina: ', data.frame.columns[-1])
print('Moguće vrijednosti izlazne veličine: ', data.frame['target'].unique())

input = data.frame.drop(columns='target')
print(input.corr())

fig, axes = plt.subplots(3, 10, figsize=(20, 10))

for ax, feature in zip(axes.flatten(), data.feature_names):
	ax.boxplot(data.frame[feature])
	ax.set_title(feature)
	ax.set_xlabel(feature)
	ax.set_ylabel('vrijednosti')

plt.tight_layout()
plt.show()

scaler = StandardScaler()
scaled = scaler.fit_transform(data.data)
scaled_df = data.data.copy()
scaled_df.loc[:, :] = scaled

fig, axes = plt.subplots(3, 10, figsize=(20, 10))

for ax, feature in zip(axes.flatten(), data.feature_names):
	ax.boxplot(scaled_df[feature])
	ax.set_title(feature)
	ax.set_xlabel(feature)
	ax.set_ylabel('skalirane vrijednosti')

plt.tight_layout()
plt.show()

#2. zadatak
X = data.frame.drop(columns='target')
y = data.frame['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_vis = X_train[['mean radius', 'mean concave points']].copy()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression().fit(X_train, y_train)

scaler_vis = StandardScaler()
X_train_vis = scaler_vis.fit_transform(X_train_vis)
model_vis = LogisticRegression().fit(X_train_vis, y_train)
plot_decision_regions(X_train_vis, y_train.to_numpy(), model_vis)

y_test_pred = model.predict(X_test)

print('Točnost: ' , accuracy_score(y_test, y_test_pred))
print('Preciznost: ', precision_score(y_test, y_test_pred))
print('Odziv: ', recall_score(y_test, y_test_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred)).plot()
plt.show()

#3. zadatak
X = data.frame.drop(columns='target')
y = data.frame['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential()
model.add(layers.Input(shape=(30,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
	keras.callbacks.TensorBoard(log_dir='logs/cancer')
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=16, validation_split=0.1)

model.fit(X_train, y_train, callbacks=callbacks, epochs=50, batch_size=160, validation_split=0.1)

model.save('2.keras')

loaded_model = load_model('2.keras')
y_test_pred = loaded_model.predict(X_test)
y_test_pred = (y_test_pred >= 0.5).astype(int)
print('Točnost: ' , accuracy_score(y_test, y_test_pred))
print('Preciznost: ', precision_score(y_test, y_test_pred))
print('Odziv: ', recall_score(y_test, y_test_pred))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred), display_labels=[0, 1]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

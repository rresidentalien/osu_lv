#imports
import keras
from keras import layers
from keras.datasets import fashion_mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

#1. zadatak
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = [
	'T-shirt/top',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot',
]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for class_id, ax in enumerate(axes.flatten()):
	sample_idx = np.where(y_train == class_id)[0][0]
	ax.imshow(X_train[sample_idx], cmap='gray')
	ax.set_title(class_names[class_id])
	ax.axis('off')

plt.tight_layout()
plt.show()

counts_first_10000 = np.bincount(y_train[:10000], minlength=10)
counts_full = np.bincount(y_train, minlength=10)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

axes[0].bar(class_names, counts_first_10000, color='steelblue')
axes[0].set_title('Distribucija klasa u prvih 10000 uzoraka')
axes[0].set_xlabel('Klasa')
axes[0].set_ylabel('Broj uzoraka')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(class_names, counts_full, color='darkorange')
axes[1].set_title('Distribucija klasa u cijelom skupu')
axes[1].set_xlabel('Klasa')
axes[1].set_ylabel('Broj uzoraka')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#2. zadatak
X_train = X_train[:10000, :]
y_train = y_train[:10000]
X_test = X_test[:10000, :]
y_test = y_test[:10000]

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = SVC(kernel='rbf', gamma='scale', C=1)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred)).plot()
plt.show()

svm_grid = {
	'C' : [0.01, 0.1],
	'gamma' : [0.1]
}

svm_search = GridSearchCV(
	estimator=SVC(kernel='rbf'),
	param_grid=svm_grid,
	cv=3,
	scoring='accuracy'
)
svm_search.fit(X_train, y_train)

print('Najbolji parametri (GridSearch): ', svm_search.best_params_)
print('Najbolja CV točnost: ', svm_search.best_score_)

best_svm = svm_search.best_estimator_
y_test_pred_best = best_svm.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred_best)).plot()
plt.title('Matrica zabune (najbolji SVM iz GridSearch)')
plt.show()


#3. zadatak
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train[:10000, :]
y_train = y_train[:10000]
X_test = X_test[:10000, :]
y_test = y_test[:10000]

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

network = keras.Sequential()
network.add(layers.Input(shape=(28,28,1)))
network.add(layers.Conv2D(32, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.summary()

network.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

callbacks1 = [
	keras.callbacks.TensorBoard(log_dir='logs/fashion1')
]

network.fit(
	X_train,
	y_train,
	epochs=5,
	batch_size=64,
	validation_split=0.1,
	callbacks=callbacks1
)

bigbatch_network = keras.Sequential()
bigbatch_network.add(layers.Input(shape=(28,28,1)))
bigbatch_network.add(layers.Conv2D(32, (3, 3), activation='relu'))
bigbatch_network.add(layers.MaxPooling2D((2, 2)))
bigbatch_network.add(layers.Conv2D(64, (3, 3), activation='relu'))
bigbatch_network.add(layers.MaxPooling2D((2, 2)))
bigbatch_network.add(layers.Flatten())
bigbatch_network.add(layers.Dense(64, activation='relu'))
bigbatch_network.add(layers.Dense(10, activation='softmax'))

bigbatch_network.summary()

bigbatch_network.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

callbacks2 = [
	keras.callbacks.TensorBoard(log_dir='logs/fashion2')
]

bigbatch_network.fit(
	X_train,
	y_train,
	epochs=5,
	batch_size=640,
	validation_split=0.1,
	callbacks=callbacks2
)

network.save('2.keras')

loaded_model = load_model('2.keras')
y_test_pred = loaded_model.predict(X_test)
y_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_classes, y_pred_classes), display_labels=class_names).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

#imports
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import keras
from keras import layers
from keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans

#1. zadatak
data = load_wine(as_frame=True)
X = data.data
y = data.target

print('Broj uzoraka: ', len(data.frame))
print('Broj značajki: ', len(data.feature_names))
print('Značajke: ', data.frame.columns[:-1])
print('Izlazna veličina: ', data.frame.columns[-1])
print('Moguće vrijednosti: ', data.frame['target'].unique())

print(data.frame.describe())

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for ax, feature in zip(axes.flatten(), X.columns):
	ax.hist(X[feature])
	ax.set_xlabel(feature)
plt.suptitle('Histogram svih značajki')
plt.show()

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
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


#2. zadatak
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_log = LogisticRegression().fit(X_train, y_train)

model_knn = KNeighborsClassifier()
grid_knn = {
	'n_neighbors' : [1, 3, 5, 7]
}
best_knn = GridSearchCV(estimator=model_knn, param_grid=grid_knn, cv=5)
best_knn.fit(X_train, y_train)

model_svm = SVC()
grid_svm = {
	'kernel' : ['linear', 'rbf']
}
best_svm = GridSearchCV(estimator=model_svm, param_grid=grid_svm, cv=5)
best_svm.fit(X_train, y_train)

print('Logistička regresija:')
y_pred_log = model_log.predict(X_test)
print('Točnost = ', accuracy_score(y_test, y_pred_log))
print('Preciznost = ', precision_score(y_test, y_pred_log, average='macro'))
print('Odziv = ', recall_score(y_test, y_pred_log, average='macro'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_log)).plot()
plt.title('Matrica zabune - logistička regresija')
plt.show()

print('KNN:')
y_pred_knn = best_knn.predict(X_test)
print('Točnost = ', accuracy_score(y_test, y_pred_knn))
print('Preciznost = ', precision_score(y_test, y_pred_knn, average='macro'))
print('Odziv = ', recall_score(y_test, y_pred_knn, average='macro'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_knn)).plot()
plt.title('Matrica zabune - KNN')
plt.show()

print('SVM:')
y_pred_svm = best_svm.predict(X_test)
print('Točnost = ', accuracy_score(y_test, y_pred_svm))
print('Preciznost = ', precision_score(y_test, y_pred_svm, average='macro'))
print('Odziv = ', recall_score(y_test, y_pred_svm, average='macro'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_svm)).plot()
plt.title('Matrica zabune - SVM')
plt.show()

inertia_values = []
cluster_range = range(1, 11)
for cluster_count in cluster_range:
	kmeans_elbow = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
	kmeans_elbow.fit(X_train)
	inertia_values.append(kmeans_elbow.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(list(cluster_range), inertia_values, marker='o')
plt.xlabel('Broj klastera')
plt.ylabel('Inercija')
plt.title('Metoda lakta za odabir broja klastera')
plt.xticks(list(cluster_range))
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
test_labels = kmeans.fit_predict(X_train)
predicted_clusters = kmeans.predict(X_test)

print('K-means na testnom skupu:')
print(confusion_matrix(y_test, predicted_clusters))


#3. zadatak
network = keras.Sequential()
network.add(layers.Input(shape=(13,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(8, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))
network.summary()

y_train_categorical = keras.utils.to_categorical(y_train, num_classes=3)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes=3)

network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
	keras.callbacks.TensorBoard(log_dir='logs/wine')
]

network.fit(X_train, y_train_categorical, epochs=50, batch_size=16, validation_split=0.1, callbacks=callbacks)

network.save('2.keras')

loaded_model = load_model('2.keras')
y_test_pred = loaded_model.predict(X_test)
y_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_classes, y_pred_classes)).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

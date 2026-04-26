'''
Iris dataset sastoji se od informacija o laticama i čašicama cvijeta irisa. Dostupan je u sklopu biblioteke scikit-learn. Za svaki uzorak postoji informacija o tome kojoj klasi pripada: Setosa, Versicolour ili Virginica.

Učitajte dataset u Python pomoću naredbi:
python
from sklearn.datasets import load_iris
data = load_iris()



Zadaci:
1. Upoznajte se s datasetom. Uočite koji su podaci ulazne, a koji izlazne veličine.
2. Razmislite o sljedećim algoritmima: linearna regresija, binarna logistička regresija, višeklasna klasifikacija logističkom regresijom, KNN, SVM, algoritam K-srednjih vrijednosti.
3. Na temelju znanja o navedenim algoritmima i svojstava promatranog podatkovnog skupa obrazložite koji od predloženih algoritama su prikladni za učenje i predviđanje klase cvijeta irisa.
4. Navedite nedostatke ostalih algoritama za ovaj specifični problem.
'''

from sklearn.datasets import load_iris
import sklearn
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
data = pd.DataFrame(data=np.c_[data.data, data.target], columns=data.feature_names + ['target'])
data['target'] = data['target'].astype(int)
print(data.head(5))
print(data.describe())

plt.scatter(x=data['petal length (cm)'], y=data.target)
plt.show()
# vidimo da je klasa 0 odvojiva, dok klase 1 i 2 imaju preklapanja u duljini latice

'''
1. 
ulazne velicine: sepal length, sepal width, petal length, petal width
izlazne velicine: vrsta iris cvijeta (0, 1, 2) - klasifikacijski problem

istrazila sam dataset, pogledala neke njegove vrijednosti, prosjeke, te jednu nasumicnu ulaznu varijablu usporedila s izlaznom varijablom u scatter plotu cisto kako bih vidjela moze li se vec odavdje razaznati neka razlika medju tri moguce klase.
'''

'''
2., 3.
višeklasna klasifikacija logističkom regresijom - moze nam biti pogodna s obzirom da se koristi za klasifikaciju vise od 2 klase, ali mozda necemo imati jasne granice izmedju podataka sto bi nam onda dalo losiji model u ovom slucaju jer ovaj pristup obavlja linearnu separaciju
KNN - moze biti pogodan za nas problem jer se moze koristiti za viseklasnu klasifikaciju, a potencijalno ce nam biti koristan jer stvara nelinearnu granicu odluke
SVM - moze biti pogodan za nas problem takodjer jer se moze koristiti za klasifikacijske probleme, 
'''

'''
4.
linearna regresija - iako se linearna regresija moze koristiti i za klasifikaciju, to nam nije najpogodniji izbor jer cemo morati iz dobivenih vrijednosti traziti granice medju klasama
binarna logistička regresija - koristi se samo za klasifikaciju izmedju dvije moguce vrijednosti izlazne varijable, stoga nam nije pogodna za ovaj problem
algoritam K-srednjih vrijednosti - u teoriji bismo ga mogli koristiti, ali nas zadatak je tipican problem nadziranog ucenja dok ce nam k-means sam traziti "uzorke" u podacima i vratiti odabrani broj grupa. cak iako mi odaberemo 3, mozda necemo dobiti tocno te klase koje mi zelimo, stoga necemo koristiti k-means
'''

# viseklasna klasifikacija
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score

X_train, X_test, y_train, y_test = train_test_split(data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']], data['target'], test_size=0.3, random_state=42)

model = LogisticRegression().fit(X_train, y_train)

y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_predicted)).plot()
plt.show()
# viseklasna klasifikacija logistickom regresijom se pokazala dobrom, jer su svi primjeri dobro klasificirani


# knn
from sklearn.neighbors import KNeighborsClassifier

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)

y_train_pred_knn5 = knn5.predict(X_train)
y_test_pred_knn5 = knn5.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred_knn5)).plot()
plt.show()
# knn je takodjer tocno klasificirao sve primjere


# svm
from sklearn import svm

svm_model = svm.SVC(kernel='poly', degree=5)
svm_model.fit(X_train, y_train)

y_test_pred_svm = svm_model.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred_svm)).plot()
plt.show()
# i svm je tocno klasificirao sve primjere


# mozemo probati i linearnu regresiju i k-means.
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)

y_test_predicted = model.predict(X_test)

plt.scatter(y_test, y_test_predicted)
plt.show()
# linearna regresija nije savrsena, ali mozemo uspjesno identificirati bez preklapanja da je setosa od -0.1 do 0.06, versicolour od 1 do 1.4 i virginica od 1.58 do 2.2

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
model_km.fit(X_train_scaled)

labels = model_km.predict(X_test_scaled)
centers = scaler.inverse_transform(model_km.cluster_centers_)

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.show()

# Upoznajte se s datasetom. Pripremite podatke za uˇcenje.
from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
import numpy as np
iris = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
true_labels = iris['target'].astype(int)
iris = iris.drop(columns='target')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris = pd.DataFrame(data=scaler.fit_transform(iris), columns=iris.columns)

# a) Prona  ̄dite optimalni broj klastera Kza klasifikaciju cvijeta irisa algoritmom K srednjih vrijednosti.
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)

# b) Grafiˇcki prikažite lakat metodu.
import matplotlib.pyplot as plt
sse = []
for k in range(2, 11):
  kmeans = KMeans(n_clusters=k).fit(iris)
  sse.append({'k': k, 'sse': kmeans.inertia_})

sse = pd.DataFrame(sse)

plt.plot(sse.k, sse.sse)
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Lakat metoda za iris dataset')
plt.show()

# c) Primijenite algoritam K srednjih vrijednosti koji  ́ce prona ́ci grupe u podatcima. Koristite vrijednot K dobivenu u prethodnom zadatku.
model.fit(iris)
labels = model.predict(iris)
centroids = model.cluster_centers_

# d) Dijagramom raspršenja prikažite dobivene klastere. Obojite ih razliˇcitim bojama (zelena, žuta i naranˇcasta). Centroide obojite crvenom bojom. Dodajte nazive osi, naziv dijagrama i legendu. Komentirajte prikazani dijagram.
plt.scatter(iris.iloc[labels == 0, 0], iris.iloc[labels == 0, 1], c='green', label='Klaster 1')
plt.scatter(iris.iloc[labels == 1, 0], iris.iloc[labels == 1, 1], c='yellow', label='Klaster 2')
plt.scatter(iris.iloc[labels == 2, 0], iris.iloc[labels == 2, 1], c='orange', label='Klaster 3')

plt.scatter(centroids[:, 0], centroids[:, 1], c='red')

plt.xlabel(iris.columns[0])
plt.ylabel(iris.columns[1])
plt.title('K-means klasteri')
plt.legend()
plt.show()

# e) Usporedite dobivene klase sa njihovim stvarnim vrijednostima. Izraˇcunajte toˇcnost klasifikacije
mapping = {} # ovdje ćemo spremati vrijednosti u obliku npr. {0: 2, 1: 1, 2: 0} oblika kmeans klaster (0,1,2): stvarna klasa (setosa 0, versicolor 1, virginica 2)
for cluster in range(3): # prolazimo kroz sve klastere
  mapping[cluster] = true_labels[labels == cluster].mode()[0] # u mapping za svaki klaster spremamo ovako: labels == broj klastera što nam daje niz true/false vrijednosti umjesto brojeva 0, 1, 2 (stvarnih vrijednosti). true_labels će od svih vrijednosti ostaviti samo one pod true (napravili smo ustvari masku), gdje bi nam trebali većinom biti brojevi samo jedne stvarne klase. npr za cluster = 0 u true_labels su nam ostali [1,1,1,1,0,2,1,1,1]. iz toga uzimamo mode() čime ćemo dobiti najčešću stvarnu klasu (to je nama naravno 1). onda ćemo samo iz tog niza uzeti baš broj s [0]. i tako smo dobili da je za kmeans klaster 0, zapravo klasa 1. ovo možemo napraviti jer smo u labels dobili array s jednako vrijednosti kao i true_labels i one su na jednakim pozicijama, znači ako je nama npr. true_labels[0] = 1, a labels[0] = 2, znamo da je ono što je kmeans označio kao klaster 2 zapravo klasa 1 tj. versicolor, samo moramo to napraviti za sve primjerke i uzeti "najčešći" jer nam naravno nisu svi primjerci dobro predviđeni
predicted_labels = pd.Series(labels).map(mapping) # na kraju ćemo samo pretvoriti brojeve iz labels (brojeve klastera) u točne brojeve klasa koristeći mapping i to spremiti u predicted_labels

from sklearn.metrics import accuracy_score
print('Točnost: ', accuracy_score(true_labels, predicted_labels))

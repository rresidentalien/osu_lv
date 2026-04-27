# Iris Dataset sastoji se od informacija o laticama i ˇcašicama tri razliˇcita cvijeta irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
iris = datasets.load_iris()

# Upoznajte se s datasetom
import pandas as pd
import numpy as np

iris = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
iris['target'] = iris['target'].astype(int)

# Podijelite ga na ulazne podatke X i izlazne podatke y predstavljene klasom cvijeta. Pripremite podatke za uˇcenje neuronske mreže (kategoriˇcke veliˇcine, skaliranje...). Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
from keras.utils import to_categorical

y = iris['target']
y = to_categorical(y, num_classes=3) # pretvaramo y u kategoričke veličine umjesto 0, 1, 2. ista stvar kao get_dummies iz pandasa, samo je ova baš iz kerasa pa ćemo koristiti ovu
X = iris.drop(columns='target')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns)

# a) Izgradite neuronsku mrežu sa sljede ́cim karakteristikama:
'''
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 7 neurona i koristi relu aktivacijsku funkciju
- tre ́ci skriveni sloj ima 5 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima 3 neurona i koristi softmax aktivacijsku funkciju.
-izme  ̄du prvog i drugog te drugog i tre ́ceg sloja dodajte Dropout sloj s 20%, odnosno 30% izbaˇcenih neurona
'''
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Ispišite informacije o mreži u terminal.
model.summary()

# b) Podesite proces treniranja mreže sa sljede ́cim parametrima:
'''
- loss argument: categorical_crossentropy
- optimizer: adam
- metrika: accuracy.
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 450) i veliˇcinom batch-a 7.
model.fit(X_train, y_train, epochs=450, batch_size=7)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
model.save('3.keras')

# e) Izvršite evaluaciju mreže na testnom skupu podataka.
from keras.models import load_model

loaded_model = load_model('3.keras')
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Gubitak: ', loss)
print('Tocnost: ', accuracy)

# f) Izvršite predikciju mreže na skupu podataka za testiranje.
y_pred = loaded_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1) # pošto nam je y one-hot-encodan, da npr. redak [0, 1, 0] pretvorimo natrag u 1

# f) Prikažite matricu zabune za skup podataka za testiranje.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_classes, y_pred_classes), display_labels=[0, 1, 2]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

# f) Komentirajte dobivene rezultate i predložite kako biste ih poboljšali, ako je potrebno.
# Gubitak:  0.056256309151649475 - iznimno mali loss, model je s velikom sigurnošću točno klasificirao primjere
# Tocnost:  0.9666666388511658 - vrlo velika točnost, čak i viša nego na skupu za učenje (94%)
# iz matrice konfuzije vidimo da je model krivo klasificirao samo jedan primjerak - jedan primjerak cvijeta versicolour je označio kao virginica, što ima smisla jer su one sličnije jedna drugoj nego setosa

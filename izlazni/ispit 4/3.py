# Uˇcitajte dane podatke u obliku numpy polja data. Podijelite ih na ulazne podatke X i izlazne podatke y. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
import numpy as np
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
real_bmi = (data[:, 5] != 0)
data = data[real_bmi]

X = data[:, :-1]
y = data[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler # ovdje se ne traži, ali pošto inače piše da se pripremi podatke, a ovdje se radi na numpy polju umjesto dataframe, napravila sam i ovdje za primjer
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# a) Izgradite neuronsku mrežu sa sljede ́cim karakteristikama:
'''
- model oˇcekuje ulazne podatke s 8 varijabli
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
'''
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Ispišite informacije o mreži u terminal.
model.summary()

# b) Podesite proces treniranja mreže sa sljede ́cim parametrima:
'''
- loss argument: cross entropy
- optimizer: adam
- metrika: accuracy.
'''
model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veliˇcinom batch-a 10.
model.fit(X_train, y_train, epochs=150, batch_size=10)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
model.save('3.keras')

# e) Izvršite evaluaciju mreže na testnom skupu podataka.
from keras.models import load_model

loaded_model = load_model('3.keras')
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Gubitak: ', loss)
print('Točnost: ', accuracy)

# f) Izvršite predikciju mreže na skupu podataka za testiranje. 
y_test_pred = loaded_model.predict(X_test, verbose=0)
y_test_pred = (y_test_pred >= 0.5).astype(int)

# Prikažite matricu zabune za skup podataka za testiranje. 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred), display_labels=[0, 1]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

# Komentirajte dobivene rezultate.
# Gubitak:  0.5454848408699036 - na skupu za učenje loss je bio oko 0.33, što znači da je model bio sigurniji na skupu za učenje
# Točnost:  0.7631579041481018 - točnost na skupu za učenje je bila 84%, što nije značajna razlika pa model vjerojatno nije overfittan
# iz matrice konfuzije vidimo da je ovaj model imao sličan problem kao i model logističke regresije. solidno je raspoznavao negativne primjerke (77 stvarno negativnih i 20 lažno pozitivnih), ali ima 39 stvarno pozitivnih i 16 lažno negativnih što je gotovo 30%. dakle model ne može baš dobro klasificirati da osoba ima dijabetes, možda zato jer neke osobe imaju indikatore dijabetesa (npr. pretilost), a zapravo nemaju dijabetes

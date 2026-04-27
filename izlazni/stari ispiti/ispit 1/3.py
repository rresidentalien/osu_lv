'''
Datoteka winequality-red.csv sadrži kemijska mjerenja crnih vina i njihovu
kvalitetu. Upoznajte se s datasetom. Više informacija nalazi se u datoteci winequality.names.
Uˇcitajte dane podatke. Podijelite skup na ulazne podatke X i izlazne podatke y predstavljene
kvalitetom vina. Zamijenite sve vrijednosti kvalitete vina manje od 6 s 0, a one koje imaju
vrijednost 6 ili ve ́cu s 1 kako biste dobili dvije izlazne klase. Podijelite podatke na skup za uˇcenje
i skup za testiranje modela u omjeru 80:20. Pripremite podatke za uˇcenje.
'''
import pandas as pd
wines = pd.read_csv('winequality-red.csv')

wines['quality'] = (wines['quality'] >= 6).astype(int)

X = wines.drop(columns='quality')
y = wines['quality']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns)
# sve značajke su nam već numeričke, stoga pod "pripremom podataka za učenje" ne moramo raditi i OHE

# a) Izgradite neuronsku mrežu sa sljede ́cim karakteristikama:
'''
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 22 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- tre ́ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
'''
from tensorflow import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Input(shape=(11,))) # 11 znacajki -> shape = 11
model.add(layers.Dense(22, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Ispišite informacije o mreži u terminal.
model.summary()

#b) Podesite proces treniranja mreže sa sljede ́cim parametrima:
'''
- loss argument: binary_crossentropy
- optimizer: adam
- metrika: accuracy.
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte s 800) i proizvoljnom veliˇcinom batch-a (pokušajte s 50).
model.fit(X_train, y_train, epochs=800, batch_size=50)

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
y_pred = (y_pred >= 0.5).astype(int)

# f) Prikažite matricu zabune za skup podataka za testiranje.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[0, 1]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

# Interpretirajte dobivene rezultate.
# Gubitak:  0.5529254674911499 - loss je bio manji na skupu za učenje (oko 0.4), što može značiti da je blago prenaučio podatke
# Tocnost:  0.753125011920929 - model ispravno klasificira 75% uzoraka, što je jednaka točnost kao i za model logističke regresije. na skupu za učenje točnost je bila oko 82%, pa i ova metrika sugerira da je model malo prenaučio podatke
# iz matrice zabune vidimo da je model imao otprilike podjednako lažno negativnih (39) i lažno pozitivnih (38) primjeraka, stoga možemo zaključiti model nije previše ili premalo selektivan, nego jednostavno nije sve primjere dobro klasificirao. ipak, model je i dalje blago overfittan, što bi se moglo popraviti ranijim završavanjem

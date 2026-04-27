'''
Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke ypredstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 75:25.
Izbacite izostale i null vrijednosti. Skalirajte podatke. 
'''
import pandas as pd
titanic = pd.read_csv('titanic.csv')

print(titanic[['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']].isnull().sum()) # nedostaju dvije vrijednosti u Embarked
titanic = titanic.dropna(subset=['Embarked'])

X = titanic[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = titanic['Survived']

X = pd.get_dummies(X, columns=['Sex', 'Embarked'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['Pclass', 'Fare']] = scaler.fit_transform(X_train[['Pclass', 'Fare']]) # skalirat ćemo samo brojčane vrijednosti
X_test[['Pclass', 'Fare']] = scaler.transform(X_test[['Pclass', 'Fare']])

# a) Izgradite neuronsku mrežu sa sljede ́cim karakteristikama:
'''
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- tre ́ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
'''
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Input(shape=(X_train.shape[1], )))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Ispišite informacije o mreži u terminal.
model.summary()

# b) Podesite proces treniranja mreže sa sljede ́cim parametrima
'''
- loss argument: binary_crossentropy
- optimizer: adam
- metrika: accuracy.
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i veliˇcinom batch-a 5.
model.fit(X_train, y_train, epochs=100, batch_size=5)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
model.save('3.keras')

# e) Izvršite evaluaciju mreže na testnom skupu podataka.
from keras.models import load_model

loaded_model = load_model('3.keras')
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Gubitak: ', loss)
print('Tocnost: ', accuracy)

# f) Izvršite predikciju mreže na skupu podataka za testiranje.
y_test_pred = loaded_model.predict(X_test, verbose=0)
y_test_pred = (y_test_pred >= 0.5).astype(int)

# Prikažite matricu zabune za skup podataka za testiranje.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred), display_labels=[0, 1]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()

# Komentirajte dobivene rezultate i predložite kako biste ih poboljšali, ako je potrebno.
# Gubitak:  0.44949448108673096 - model nije bio baš najsigurniji, ali loss je sličan onome na skupu za učenje, pa znamo da nije prenaučio podatke jer nije pravio velike greške u procjeni u odnosu na skupu za učenje
# Tocnost:  0.8116592168807983 - 81% primjeraka je točno klasificirano, što je solidan rezultat, također jako sličan skupu za učenje što je dobro
# iz matrice konfuzije vidimo da je model imao 18 lažno pozitivnih i 27 lažno negativnih primjeraka, stoga je model možda prestrogo određivao je li putnik preživio i dosta stvarno pozitivnih primjeraka klasificirao kao negativne

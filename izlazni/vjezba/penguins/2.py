'''
- Zadaci uključuju:
  - Predobradu i analizu podataka
  - Implementaciju algoritama strojnog učenja
  - Provedbu učenja i testiranja modela
'''

import pandas as pd
data = pd.read_csv('penguins.csv')

# predobrada
data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

X = data.drop(columns=['species', 'sex']) # species je izlazna varijabla, spol nije bitan za određivanje species
y = data['species']

X = pd.get_dummies(X, columns=['island'])
y = y.map({
  'Adelie': 0,
  'Chinstrap': 1,
  'Gentoo': 2,
}).astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# višeklasna klasifikacija logističkom regresijom
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
print('Train:')
y_train_pred = model.predict(X_train)
print('Tocnost: ', accuracy_score(y_train, y_train_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred)).plot()
plt.show()

print('Test:')
y_test_pred = model.predict(X_test)
print('Tocnost: ', accuracy_score(y_test, y_test_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred)).plot()
plt.show()

# knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

print('Train:')
y_train_pred = model.predict(X_train)
print('Tocnost: ', accuracy_score(y_train, y_train_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred)).plot()
plt.show()

print('Test:')
y_test_pred = model.predict(X_test)
print('Tocnost: ', accuracy_score(y_test, y_test_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred)).plot()
plt.show()

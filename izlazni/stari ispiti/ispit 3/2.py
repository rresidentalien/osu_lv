# Uˇcitajte dane podatke.
import pandas as pd
titanic = pd.read_csv('titanic.csv')

# Izbacite izostale i null vrijednosti.
print(titanic[['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']].isnull().sum()) # nedostaju dvije vrijednosti u Embarked
titanic = titanic.dropna(subset=['Embarked'])

# Podijelite ih na ulazne podatke X predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke ypredstavljene stupcem Survived.
X = titanic[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = titanic['Survived']

# s obzirom da nam za knn trebaju sve brojčane vrijednosti, još ćemo napraviti OHE na stupcima Sex i Embarked:
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])

# Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Skalirajte podatke.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['Pclass', 'Fare']] = scaler.fit_transform(X_train[['Pclass', 'Fare']]) # skalirat ćemo samo brojčane vrijednosti
X_test[['Pclass', 'Fare']] = scaler.transform(X_test[['Pclass', 'Fare']])

# a) Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Vizualizirajte podatkovne primjere i granicu odluke.
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
model_vis = KNeighborsClassifier(n_neighbors=5).fit(X_train[['Pclass', 'Fare']], y_train)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02): # ova funkcija je iz predloška za LV6 (knn i svm): "Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija plot_decision_regions."
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
                    label=cl)
plot_decision_regions(X_train[['Pclass', 'Fare']].to_numpy(), y_train.to_numpy(), classifier=model_vis) # funkcija radi samo za dve značajke i treba joj numpy array, a ne pandas series
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Podatkovni primjeri i granica odluke - KNN')
plt.show()

# b) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Komentirajte dobivene rezultate.
from sklearn.metrics import accuracy_score

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('Točnost na skupu za učenje:', accuracy_score(y_train, y_train_pred)) # 0.8537 - dobra vrijednost, biranjem drugog K bi točnost mogla porasti
print('Točnost na skupu za testiranje:', accuracy_score(y_test, y_test_pred)) # 0.8052 - dovoljno slično točnosti na skupu za učenje -> K nije toliko mali da je model prenaučio podatke

# c) Pomo ́cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN.
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [1, 3, 4, 5, 7, 10, 20, 30, 50, 100]}
optimal_model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring ='accuracy').fit( X_train, y_train) # 'unakrsna validacija' => radimo GridSearchCV.fit()
print('Optimalni K:', optimal_model.best_params_['n_neighbors'])
print('Najbolja točnost:', optimal_model.best_score_)

# d) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje za dobiveni K. Usporedite dobivene rezultate s rezultatima kada je K=5.
y_test_pred_optimal = optimal_model.predict(X_test)
print('Točnost prvog modela: ', accuracy_score(y_test, y_test_pred))
print('Točnost optimalnog modela: ', accuracy_score(y_test, y_test_pred_optimal))
# optimalni model nam je zapravo također bio model s K=5, pa smo dobili isti rezultat

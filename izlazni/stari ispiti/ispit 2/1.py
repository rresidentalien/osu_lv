# Iris Dataset sastoji se od informacija o laticama i ˇcašicama tri razliˇcita cvijeta irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
iris = datasets.load_iris()

# Upoznajte se s datasetom
import pandas as pd
import numpy as np

iris = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
iris['target'] = iris['target'].astype(int)
print(iris.head(5))
print(iris.describe())

# a) Prikažite odnos duljine latice i ˇcašice svih pripadnika klase Virginica pomo ́cu scatter dijagrama zelenom bojom.
import matplotlib.pyplot as plt

plt.scatter(x=iris[iris['target'] == 2]['petal length (cm)'], y=iris[iris['target'] == 2]['sepal length (cm)'], c='green', label='Virginica')

# a) Dodajte na isti dijagram odnos duljine latice i ˇcašice svih pripadnika klase Setosa, sivom bojom.
plt.scatter(x=iris[iris['target'] == 0]['petal length (cm)'], y=iris[iris['target'] == 0]['sepal length (cm)'], c='gray', label='Setosa')

# a) Dodajte naziv dijagrama i nazive osi te legendu.
plt.title('Usporedba duljine latice i čašice za klase Virginica i Setosa')
plt.xlabel('Duljina latice (cm)')
plt.ylabel('Duljina čašice (cm)')
plt.legend()
plt.show()

# a) Komentirajte prikazani dijagram.
# iz dijagrama raspršenja možemo vidjeti da primjerci klase Virginica imaju znatno dulje latice i većina primjeraka ima dulju čašicu. iako samo prema duljini čašice ove dvije klase nisu linearno odvojive, razlika u duljini latice je dovoljno velika da možemo jasno raspoznati koja je koja klasa.

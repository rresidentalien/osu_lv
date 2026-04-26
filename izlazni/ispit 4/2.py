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

# a) Izgradite model logistiˇcke regresije pomo ́cu scikit-learn biblioteke na temelju skupa podataka za uˇcenje
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)

# b) Provedite klasifikaciju skupa podataka za testiranje pomo ́cu izgra  ̄denog modela logistiˇcke regresije.
y_test_pred = model.predict(X_test)

# c) Izraˇcunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezultate.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred), display_labels=[0, 1]).plot()
plt.title('Matrica zabune za testni skup')
plt.show()
# model je točno klasificirao veliki broj stvarno negativnih primjeraka (82), ali samo 33 stvarno pozitivna primjerka. imao je 22 lažno negativna i 15 lažno pozitivnih primjeraka, pa bismo mogli zaključiti da je model "prestrog" i da znatan broj stvarno pozitivnih primjeraka ne prepoznaje kao pozitivne.

# d) Izraˇcunajte toˇcnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte dobivene rezultate
from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Točnost: ', accuracy_score(y_test, y_test_pred)) # Točnost:  0.756578947368421 - točnost od 75% nije osobito visoka; model 1/4 primjeraka pogrešno klasificira
print('Preciznost: ', precision_score(y_test, y_test_pred)) # Preciznost:  0.6875 - model znatan dio stvarno negativnih primjeraka označava kao pozitivne
print('Odziv: ', recall_score(y_test, y_test_pred)) # Odziv:  0.6 - model čak 40% stvarno pozitivnih primjeraka nije prepoznao
# i iz ovih metrika kao i iz matrice zabune vidimo da model nije osobito točan

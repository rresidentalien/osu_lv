# Uˇcitajte dane podatke.
import pandas as pd

wines = pd.read_csv('winequality-red.csv')

# Zamijenite sve vrijednosti kvalitete vina manje od 6 s 0, a one koje imaju vrijednost 6 ili ve ́cu s 1 kako biste dobili dvije izlazne klase
wines['quality'] = (wines['quality'] >= 6).astype(int)

# Podijelite skup na ulazne podatke X i izlazne podatke y predstavljene kvalitetom vina.
X = wines.drop(columns='quality')
y = wines['quality']

# Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizirajte podatke.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns)

# a) Izgradite linearni regresijski model. Ispišite parametre modela.
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)

parameters = ''
for feature, coef in zip(X_train.columns, model.coef_):
    parameters += f'{coef} * {feature} + '
parameters += model.intercept_.astype(str)
print('parameters: ', parameters)

# b) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje.
y_test_predicted = model.predict(X_test)

# b) Prikažite pomo ́cu dijagrama raspršenja odnos izme  ̄du stvarnih vrijednosti izlazne veliˇcine i procjene dobivene modelom.
import matplotlib.pyplot as plt
plt.scatter(x=y_test, y=y_test_predicted)
plt.xlabel('Stvarni podaci')
plt.ylabel('Predviđeni podaci')
plt.title('Odnos stvarnih vrijednosti i procjene modela')
plt.show()

# b) Interpretirajte dobivene rezultate.
# na dijagramu raspršenja vidimo da su naše stvarne vrijednosti na x osi 0 ili 1, kako smo i ranije zamijenili. na y osi, vrijednosti se kreću između 1.2 i -0.1. idealan model bi sve uzorke stvarne vrijednosti 0 smjestio na 0 i na y osi, te uzorke stvarne vrijednosti 1 također na 1 na y osi. međutim, na ovom modelu za uzorke stvarne vrijednosti 0 predviđene vrijednosti kreću se od -0.1 do 0.9, a za uzorke stvarne vrijednosti 1 predviđene vrijednosti kreću se od 0.02 do 1.2.

# c) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika (RMSE, MAE, MAPE i R2) na skupu podataka za testiranje. Interpretirajte dobivene rezultate.
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

print('RMSE: ', root_mean_squared_error(y_test, y_test_predicted))
# RMSE:  0.41990367006419804 - s obzirom da nama vrijednosti y idu od 0 do 1, ovo je dosta visok RMSE. znači da se u prosjeku predviđena vrijednost udaljava od stvarne za ~0.42
print('MAE: ', mean_absolute_error(y_test, y_test_predicted))
# MAE:  0.3609984013299049 - MAE promatra apsolutno odstupanje umjesto kvadratnog, čime ne kažnjava veća odstupanja kao RMSE i MSE. s obzirom da je manji od RMSE, možemo zaključiti da postoje neka veća odstupanja u predviđenim vrijednostima
print('MAPE: ', mean_absolute_percentage_error(y_test, y_test_predicted))
# MAPE:  731720272290897.4 - dobili smo jako veliki MAPE zato što su nam izlazne vrijednosti 0 i 1 i pri tako malim vrijednostima MAPE može biti 'lažno' prevelik
print('R2: ', r2_score(y_test, y_test_predicted))
# R2:  0.28463587961256753 - samo 28% varijacije u podacima je obuhvaćeno ovim modelom, što nam je još jedan pokazatelj da model ne procjenjuje baš dobro izlaznu veličinu

# ne traži se, ali kako bih mogla usporediti rezultat s idućim zadatkom:
y_test_predicted = (y_test_predicted >= 0.5).astype(int)
from sklearn.metrics import accuracy_score
print('Tocnost: ', accuracy_score(y_test, y_test_predicted))
# Tocnost:  0.75

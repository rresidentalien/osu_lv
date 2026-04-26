# a) Za koliko žena postoje podatci u ovom skupu podataka?
import pandas as pd

titanic = pd.read_csv('titanic.csv')
print('Broj žena: ', len(titanic[titanic['Sex'] == 'female']))

# b) Koliki postotak osoba nije preživio potonu ́ce broda
print('Postotak osoba koji nije preživio: {:.2f}%'.format(len(titanic[titanic['Survived'] == 0]) / len(titanic) * 100))

# c) Pomo ́cu stupˇcastog dijagrama prikažite postotke preživjelih muškaraca (zelena boja) i žena (žuta boja). Dodajte nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka preživljavanja.
import matplotlib.pyplot as plt

survived_women = len(titanic[(titanic['Sex'] == 'female') & (titanic['Survived'] == 1)]) / len(titanic[titanic['Sex'] == 'female']) * 100
survived_men = len(titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 1)]) / len(titanic[titanic['Sex'] == 'male']) * 100

plt.bar(['Žene', 'Muškarci'], [survived_women, survived_men], color=['yellow', 'green'])
plt.xlabel('Spol')
plt.ylabel('Postotak preživjelih (%)')
plt.title('Postotak preživljavanja prema spolu')
plt.show()
# s obzirom da znamo da su se na brodove za spašavanje prvo ukrcavali žene i djeca, to objašnjava činjenicu da je preživjelo preko 70% žena i samo ispod 20% muškaraca. postoji visoka korelacija između spola i postotka preživljavanja.

# d) Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih muškaraca?
average_women = titanic[(titanic['Sex'] == 'female') & (titanic['Survived'] == 1)]['Age'].mean()
average_men = titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 1)]['Age'].mean()
print(f'Prosjecna dob svih preživjelih žena: {average_women:.2f}')
print(f'Prosjecna dob svih preživjelih muškaraca: {average_men:.2f}')

# e) Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte.
print('1. klasa: ', titanic[(titanic['Sex'] == 'male') & (titanic['Pclass'] == 1) & (titanic['Survived'] == 1)]['Age'].max()) # 80
print('2. klasa: ', titanic[(titanic['Sex'] == 'male') & (titanic['Pclass'] == 2) & (titanic['Survived'] == 1)]['Age'].max()) # 62
print('3. klasa: ', titanic[(titanic['Sex'] == 'male') & (titanic['Pclass'] == 3) & (titanic['Survived'] == 1)]['Age'].max()) # 45
# s obzirom da su karte 3. klase bile jeftinije, s jedne strane možemo pretpostaviti da su u 3. klasi bili većinom mlađi putnici, stoga je i niža dob preživjelih. s druge strane, u 1. klasi su preživjeli i stariji putnici jer su vjerojatno bili bliže brodovima za spašavanje, a i možda su imali prednost nad siromašnijim putnicima.

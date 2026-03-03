'''
Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py uˇcitava dane podatke u obliku numpy polja data pri ˇcemu je u
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a tre´ci
stupac polja je masa u kg.

a) Na temelju veliˇcine numpy polja data, na koliko osoba su izvršena mjerenja?
b) Prikažite odnos visine i mase osobe pomo´cu naredbe matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izraˇcunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1)
'''

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=",", skiprows=1)

# a)
count = data.shape[0]
print("Broj osoba: ", count)

# b)
heights = data[:, 1]
weights = data[:, 2]

plt.scatter(heights, weights)
plt.xlabel("Visina [cm]")
plt.ylabel("Masa [kg]")
plt.title("Odnos visine i mase")
plt.show()

# c)
plt.scatter(heights[1:count:50], weights[1:count:50])
plt.xlabel("Visina [cm]")
plt.ylabel("Masa [kg]")
plt.title("Odnos visine i mase svake 50. osobe")
plt.show()

# d)
print("Najmanja visina: ", np.min(heights))
print("Najveća visina: ", np.max(heights))
print("Prosječna visina: ", np.mean(heights))

# e)
male = (data[:, 0] == 1)
female = (data[:, 0] == 0)

print("Muškarci:")
print("Minimalna visina: ", np.min(heights[male]))
print("Maksimalna visina: ", np.max(heights[male]))
print("Prosječna visina: ", np.mean(heights[male]))

print("Žene:")
print("Minimalna visina: ", np.min(heights[female]))
print("Maksimalna visina: ", np.max(heights[female]))
print("Prosječna visina: ", np.mean(heights[female]))

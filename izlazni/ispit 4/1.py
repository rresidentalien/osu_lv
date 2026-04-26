# Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu otkrivanja dijabetesa, pri ˇcemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes). Uˇcitajte dane podatke u obliku numpy polja data.
import numpy as np
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# a) Na temelju veliˇcine numpy polja data, na koliko osoba su izvršena mjerenja?
print(f'Mjerenje je izvršeno na {data.shape[0]} osoba')

# b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
age = data[:, 7]
bmi = data[:, 5]

print(f'NaN vrijednosti u Age: {np.isnan(age).sum()}')
print(f'NaN vrijednosti u BMI: {np.isnan(bmi).sum()}')

mask = ~(np.isnan(age) | np.isnan(bmi)) # ionako nema null vrijednosti ali ostavit ću ovdje za primjer
data = data[mask]
data = np.unique(data, axis=0)

real_bmi = (data[:, 5] != 0)
data = data[real_bmi]

print(f'Preostalo je {data.shape[0]} uzoraka')

# c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomo ́cu scatter dijagrama. Dodajte naziv dijagrama i nazive osi s pripadaju ́cim mjernim jedinicama. Komentirajte odnos dobi i BMI prikazan dijagramom.
import matplotlib.pyplot as plt
age = data[:, 7]
bmi = data[:, 5]
plt.scatter(x=age, y=bmi)
plt.xlabel('Age (years)')
plt.ylabel('Body mass index (weight in kg/(height in m)^2)')
plt.title('Odnos dobi i BMI')
plt.show()
# iz ovog dijagrama nema jasne jake korelacije između dobi i BMI, izgleda kao da se BMI blago smanjuje porastom dobi, ali ima i manje uzoraka veće dobi

# d) Izraˇcunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne mase (BMI) u ovom podatkovnom skupu.
print('Najmanji BMI: ', np.min(bmi))
print('Najveći BMI: ', np.max(bmi))
print('Srednja vrijednost BMI: ', np.mean(bmi))

# e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one kojima nije.
diabetes = (data[:, 8] == 1) # daje niz t/f vrijednosti
bmi_no_diabetes = bmi[~diabetes] # "ugasi" sve retke bmi stupca gdje je u diabetes stupcu true
bmi_diabetes = bmi[diabetes] # "ugasi" sve retke bmi stupca gdje je u diabetes stupcu false

print('Osobe bez dijabetesa:')
print('Najmanji BMI: ', np.min(bmi_no_diabetes))
print('Najveći BMI: ', np.max(bmi_no_diabetes))
print('Srednja vrijednost BMI: ', np.mean(bmi_no_diabetes))

print('Osobe s dijabetesom:')
print('Najmanji BMI: ', np.min(bmi_diabetes))
print('Najveći BMI: ', np.max(bmi_diabetes))
print('Srednja vrijednost BMI: ', np.mean(bmi_diabetes))

# Kolikom je broju ljudi dijagonosticiran dijabetes?
print('Broj osoba s dijabetesom: ', np.sum(diabetes))

# Komentirajte dobivene vrijednosti.
# osobe s dijabetesom imaju u prosjeku veći bmi (30 naprama 35). osoba s najvećim bmi ima dijabetes, a osoba s najmanjim bmi nema dijabetes. to ima smisla jer znamo da je debljina jedan od simptoma dijabetesa

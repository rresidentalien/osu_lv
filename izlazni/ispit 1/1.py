# a) Na koliko je vina provedeno mjerenje?
import pandas as pd

wines = pd.read_csv('winequality-red.csv')
print(len(wines)) # 1599 vina

#b) Prikažite histogramom distribuciju alkoholne jakosti (alcohol). Sami izaberite broj binova. Dodajte naziv dijagrama i nazive osi. Interpretirajte rezultate prikazane dijagramom.
import matplotlib.pyplot as plt
plt.hist(wines['alcohol'], bins=20)
plt.title('Histogram distribucija jačine alkohola u vinu')
plt.xlabel('Alkohol')
plt.ylabel('Broj')
plt.show() #distribucija alkohola u vinu nije normalna, nego ima skewness prema lijevo. Najzastupljeniji postotak je oko 9.5.

# c) Koliki broj uzoraka vina ima kvalitetu manju od 6, a koliki ima 6 i veću?
print('Broj vina kvalitete manje od 6: ', len(wines[wines['quality'] < 6]))
print('Broj vina kvalitete vece ili jednake 6: ', len(wines[wines['quality'] >= 6]))

# d) Izraˇcunajte i prikažite korelaciju svih veliˇcina dostupnih u datasetu. Interpretirajte rezultate.
print(wines.corr().to_string())
# fixed acidity ima visoku korelaciju (0.67) sa citric acid, što ima smisla jer što je više kiseline, to je kiselost veća.. također što je fixed acidity viši, to je pH vrijednost niža (-0.68 korelacija)
# volatile acidity je u negativnoj korelaciji sa citric acid (-0.55) te blagoj negativnoj korelaciji s kvalitetom (-0.39)
# chlorides su u blagoj pozitivnoj korelaciji s sulphates (0.37)
# kvaliteta vina najviše pozitivno korelira s postotkom alkohola (0.47), a negativno s volatile acidity (-0.39)
# nijedna znacajka nema izrazito jaku korelaciju s kvalitetom - kvaliteta vjerojatno ovisi o vecem broju znacajki

# Cheatsheet

---

## 1) Analiza dataseta

## 1.1 Kratki opisi datasetova koji se pojavljuju u zadacima

### Iris
- 150 uzoraka, 4 numericke znacajke.
- Cilj: 3 klase (Setosa, Versicolour, Virginica) -> viseklasna klasifikacija.
- Tipicno nema null vrijednosti.
- Setosa je obicno lako odvojiva, Versicolour/Virginica djelomicno se preklapaju.

### Wine Quality (red)
- 1599 uzoraka, kemijske znacajke vina.
- Izlazna varijabla quality je ordinalna; u zadacima se cesto binarizira (<6 i >=6).
- Sve su znacajke numericke, nema null vrijednosti.
- Klase mogu biti neuravnotezene.

### Titanic
- Cilj: Survived (0/1) -> binarna klasifikacija.
- Mjesovite znacajke (npr. Pclass/Fare numericke, Sex/Embarked kategoricke).
- Ceste null vrijednosti.
- Potrebna kodiranja kategorickih varijabli (npr. one-hot).

### Pima Indians Diabetes
- Cilj: 0/1 -> binarna klasifikacija.
- Numericke znacajke.
- U praksi su neke nule zapravo nemoguce mjerne vrijednosti (npr. BMI=0), pa ih tretiras kao pogresne/missing.
- Cesta potreba za skaliranjem i analizom precision/recall trade-offa.

Korisna stranica: [Datasets description](https://inria.github.io/scikit-learn-mooc/appendix/datasets_intro.html)

---

## 2) Kako odabrati algoritam

### 2.1 Najbrze pravilo
- Kontinuirani cilj -> linearna regresija.
- Diskretna klasa 0/1 -> binarna logisticka regresija, KNN, SVM.
- Vise klasa -> multinomijalna logisticka regresija, KNN, SVM.
- Bez oznaka klasa -> K-means.

### 2.2 Tablica: kada koristiti koji algoritam

| Algoritam | Tip problema | Kada je dobar izbor | Prednosti | Nedostaci |
|---|---|---|---|---|
| Linearna regresija | Regresija | Odnos X-y je priblizno linearan, treba interpretabilan model | Brza, jednostavna, interpretabilna | Nije za klasifikaciju, osjetljiva na outliere i nelinearnost |
| Binarna logisticka regresija | Binarna klasifikacija | Treba jednostavan i stabilan baseline, odnos je priblizno linearan u logit prostoru | Interpretabilna, brza, daje vjerojatnosti | Slabija za jako nelinearne granice |
| Multiklasna logisticka regresija | Viseklasna klasifikacija | Vise klasa i treba jasan baseline | Brza, jednostavna, lako objasniti | Granice su linearne, moze podbaciti kod kompleksnih odnosa |
| KNN | Klasifikacija/regresija | Malo do srednje podataka, lokalna struktura je bitna | Jednostavan, nema klasicnog treniranja | Sporija predikcija, osjetljiv na skalu i sum, treba birati K |
| SVM | Binarna/viseklasna klasifikacija | Manji/srednji skupovi, treba robustna granica odluke | Jaka generalizacija, radi i za nelinearne granice (kernel) | Osjetljiv na odabir C/gamma/kernel, manje interpretabilan |
| K-means | Klasteriranje (bez labela) | Treba grupirati podatke i traziti prirodne skupine | Brz i jednostavan | Treba unaprijed K, osjetljiv na skalu/outliere, pretpostavlja "kuglaste" klastere |
| Neuronska mreza (MLP) | Klasifikacija/regresija | Slozeniji nelinearni odnosi, dovoljno podataka | Fleksibilna, moze dati najbolju tocnost | Treba vise podesavanja, sklonija overfitu, slabija interpretabilnost |

### 2.3 Kako argumentirati zasto neki algoritam NIJE prikladan
- Linearna regresija nije dobar izbor za klasifikaciju jer predvida kontinuirane vrijednosti i ne modelira klasu kao vjerojatnost klase.
- Binarna logisticka regresija nije prikladna kad je ciljna varijabla kontinuirana.
- K-means nije prikladan kad imas oznacene klase i trebaju ti supervised metrike, jer je to nesupervizirani algoritam.
- KNN/SVM mogu biti los izbor bez skaliranja zbog razlicitih raspona znacajki.

---

## 3) Predobrada

### 3.1 Redoslijed
1. Odradi osnovni pregled podataka.
2. Ocisti null/missing i duplicate uzorke.
3. Definiraj X i y.
4. Kodiraj kategoricke varijable (one-hot).
5. Train/test split.
6. Skaliranje (fit samo na train, transform train+test).
7. Treniranje modela.

### 3.2 Prakticna pravila
- Skaliranje je gotovo uvijek potrebno za KNN, SVM i neuronske mreze.
- Kod linearne/logisticke regresije skaliranje nije uvijek obavezno, ali cesto pomaze numerickoj stabilnosti.
- One-hot encoding radi se samo nad ulaznim varijablama, nikad nad ciljnom (osim one-hot za izlaz NN kod multiklase).
- Paziti na data leakage: nista iz testa ne smije ulaziti u fit predobrade/modela.

---

## 4) Metrike, testiranje i validacija

### 4.1 Klasifikacija
- Accuracy: ukupan postotak tocnih predikcija.
- Precision: od svih predvidenih pozitivnih, koliko je stvarno pozitivnih.
- Recall: od svih stvarno pozitivnih, koliko ih je model nasao.
- F1: balans precision i recall.
- Confusion matrix: TP, TN, FP, FN i gdje model grijesi.

### 4.2 Regresija
- MAE: prosjecna apsolutna greska.
- RMSE: jace kaznjava velike greske. Ako je puno veci od MAE, znamo da postoje outlieri ili nekoliko velikih gresaka.
- MAPE: apsolutna postotna greska, ne radi dobro za vrijednosti blizu nule.
- R2: objasnjena varijanca (blize 1 je bolje).

### 4.3 Klasteriranje
- Elbow metoda (SSE/inertia) za izbor K.
- Usporedba klastera sa stvarnim klasama (ako ih imas samo za evaluaciju).

### 4.4 Dobra praksa za ispit
- Uvijek navedi metrike na train i test.
- Ako biras hiperparametre (npr. K u KNN), koristi unakrsnu validaciju na train skupu.
- Konacnu procjenu modela radi na test skupu koji nije koristen za odabir hiperparametara.

---

## 5) Analiza rjesenja i osvrt na rezultate (drugi dio, faza 2)

### 5.1 Minimalni okvir odgovora
1. Sto model radi dobro.
2. Gdje model grijesi (tipovi gresaka iz confusion matrice ili velike regresijske greske).
3. Je li prisutan overfitting/underfitting.
4. Sto bi bio sljedeci korak poboljsanja.

### 5.2 Kako prepoznati overfitting i underfitting
- Overfitting: puno bolji rezultat na train nego na test.
- Underfitting: los rezultat i na train i na test.

### 5.3 Kako interpretirati confusion matricu
- Puno FN: model ne prepoznaje pozitivne primjere (problematicno u medicini).
- Puno FP: model cesto daje lazne alarme.
- Biraj threshold prema cijeni pogreske (sto je gore: FN ili FP).

### 5.4 Sto predloziti za poboljsanje (ovisno o rezultatima)
- Poboljsati predobradu:
	- bolja imputacija missing vrijednosti
	- uklanjanje outliera
	- bolji feature engineering
- Rjesavati neuravnotezenost:
	- class_weight
	- oversampling/undersampling
- Podesiti hiperparametre:
	- K u KNN
	- C/gamma u SVM
	- regularizacija u logistickoj regresiji
	- broj slojeva/neurona, dropout, early stopping u NN
- Probati drugi model i usporediti iste metrike.
- Koristiti validaciju i paziti na leakage.

---

## 6) Brzi "govorni" predlozak za prvi dio

"Ovo je [tip problema] jer je ciljna varijabla [tip y]. Ulazne varijable su [navesti], dataset ima [broj] uzoraka i [broj] znacajki, a postoje/ne postoje missing vrijednosti u [stupci]. Raspon varijabli je [kratko], pa je [potrebno/nije potrebno] skaliranje. Za ovaj problem najprikladniji je [algoritam] jer [razlog 1] i [razlog 2]. [Drugi algoritam] nije prikladan jer [konkretan razlog], a [treci algoritam] jer [konkretan razlog]." 

---

## 7) Moguce greske na ispitu
- Zamjena tipa problema (npr. regresija na klasifikacijskom cilju).
- One-hot nad cijelim datasetom tako da slucajno ukljucis i cilj u ulaz.
- Skaliranje prije train/test splita (data leakage).
- Ocjenjivanje samo accuracy metrikom kod neuravnotezenih klasa.
- Zakljucak bez osvrta na konkretne greske modela.

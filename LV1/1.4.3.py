'''
Napišite program koji od korisnika zahtijeva unos brojeva u beskonaˇcnoj petlji
sve dok korisnik ne upiše „ Done “ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na naˇcin da program zanemari taj unos i ispiše odgovaraju´cu poruku.
'''

lista = []

while True:
    broj = input("Unesi broj: ")
    if broj == "Done":
        break
    try:
        broj = float(broj)
        lista.append(broj)
    except:
        print("To nije broj !")

print(lista)
print("Duljina liste: ", len(lista))
print("Srednja vrijednost: ", sum(lista) / len(lista))
print("Minimalna vrijednost: ", min(lista))
print("Maksimalna vrijednost: ", max(lista))
lista.sort()
print("Sortirana lista: ", lista)

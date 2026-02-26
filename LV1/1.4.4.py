'''
Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva song.txt .
Potrebno je napraviti rjeˇcnik koji kao kljuˇceve koristi sve razliˇcite rijeˇci koje se pojavljuju u
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijeˇc (kljuˇc) pojavljuje u datoteci.
Koliko je rijeˇci koje se pojavljuju samo jednom u datoteci? Ispišite ih.
'''

dict = {}

file = open('song.txt')
for line in file:
    line = line.rstrip().replace(',', ' ')
    words = line.split()
    for word in words:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1

unique = []
for word in dict:
    if dict[word] == 1:
        unique.append(word)

print("Broj rijeci koje se pojavljuju samo jednom: ", len(unique))
print("Rijeci:", unique)

file.close()

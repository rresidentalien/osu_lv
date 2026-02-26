'''
Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri ˇcemu su neke oznaˇcene kao spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!

a) Izraˇcunajte koliki je prosjeˇcan broj rijeˇci u SMS porukama koje su tipa ham, a koliko je
prosjeˇcan broj rijeˇci u porukama koje su tipa spam.
b) Koliko SMS poruka koje su tipa spam završava uskliˇcnikom ?
'''

# a)
file = open("SMSSpamCollection.txt")

ham = 0
spam = 0
ham_words = 0
spam_words = 0

for line in file:
    line = line.rstrip().replace(',', ' ')
    words = line.split()
    if line.startswith("ham"):
        ham += 1
        ham_words += words
    if line.startswith("spam"):
        spam += 1
        spam_words += words

print("Prosjecan broj rijeci u porukama tipa ham: ", ham_words / ham)
print("Prosjecan broj rijeci u porukama tipa spam: ", spam_words / spam)

# b)
exc = 0
for line in file:
    if line.endswith("!") and line.startswith("spam"):
        exc += 1

print("Broj spam poruka koje pocinju s !: ", exc)

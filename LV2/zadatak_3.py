'''
Skripta zadatak_3.py uˇcitava sliku ’road.jpg ’. Manipulacijom odgovaraju´ce
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu ˇcetvrtinu slike po širini,
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku.
'''

import numpy as np
import matplotlib.pylab as plt

img = plt.imread("road.jpg")
img = img[:,:,0].astype(np.float32)

# a)
brighter = np.clip(img + 150, 0, 255)
plt.figure()
plt.imshow(brighter, cmap="gray")
plt.title("Posvijetljena slika")

# b)
height, width = img.shape
second_quarter = img[:,width//4:width//2]
plt.figure()
plt.imshow(second_quarter, cmap="gray")
plt.title("Druga četvrtina slike po širini")

# c)
rotated = np.rot90(img, k=-1)
plt.figure()
plt.imshow(rotated, cmap="gray")
plt.title("Zarotirana slika")

# d)
mirrorred = np.fliplr(img)
plt.figure()
plt.imshow(mirrorred, cmap="gray")
plt.title("Zrcaljena slika")
plt.show()

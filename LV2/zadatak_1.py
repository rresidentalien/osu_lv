'''
Pomo´cu funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py . Igrajte se sa slikom, promijenite boju linija, debljinu linije i
sl.
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 3, 2, 1])
y = np.array([1, 1, 2, 2, 1])

plt.plot(x, y, 'r', linewidth=1.5, marker=".", markersize=5)
plt.axis([0, 4, 0, 4])
plt.show()


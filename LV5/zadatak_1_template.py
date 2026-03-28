import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='viridis', marker='o', label='Train')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='viridis', marker='x', label='Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.colorbar(label='Klasa')
plt.title('a) Podaci iz skupa za treniranje i testiranje')
plt.show()

# b
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
y_test_p = LogRegression_model.predict(X_test)

# c
theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_.T
b = -theta0/theta2
a = -theta1/theta2
x = np.array([-5, 5])
y = a*x + b

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='viridis', marker='o')
plt.plot(x,y)
plt.fill_between(x, y, -5, color='yellow', alpha=0.2)
plt.fill_between(x, y, 5, color='blue', alpha=0.2)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('c) Granica odluke i podaci za ucenje')
plt.show()

# d
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.title('d) Matrica zabune na testnim podacima')
plt.show()
print(classification_report(y_test , y_test_p))

# e
correct = (y_test == y_test_p)
wrong = (y_test != y_test_p)

plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Dobro klasificirani')
plt.scatter(X_test[wrong, 0], X_test[wrong, 1], c='black', label='Pogresno klasificirani')
plt.plot(x,y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('e) Rezultati klasifikacije na testnom skupu')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a)
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=15)
plt.title('a) Emisija CO2 plinova')
plt.show()

# b)
data['Fuel Type'] = data['Fuel Type'].astype('category')
data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c=data['Fuel Type'].map({'X': 'black', 'D': 'blue', 'Z': 'green', 'E': 'teal'}))
plt.title('b) Odnos gradske potrosnje i emisije CO2 plinova')
plt.show()

# c)
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('c) Razdioba vangradske potrosnje s obzirom na tip goriva')
plt.show()

# d)
data.groupby('Fuel Type').size().plot(kind='bar')
plt.title('d) Broj vozila po tipu goriva')
plt.show()

# e)
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.title('e) Prosjecna emisija CO2 prema broju cilindara')
plt.show()

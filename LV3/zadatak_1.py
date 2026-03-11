import pandas as pd

# a)
data = pd.read_csv('data_C02_emission.csv')
print(len(data))
print(data.info())
data.dropna(axis=0)
data.drop_duplicates()
print(len(data))

# b)
print(data[['Make','Model','Fuel Consumption City (L/100km)']].sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3))
print(data[['Make','Model','Fuel Consumption City (L/100km)']].sort_values(by='Fuel Consumption City (L/100km)', ascending=True).head(3))

# c)
data['Engine Size (L)'] = data['Engine Size (L)'].astype(str).str.replace(",", ".")
data['Engine Size (L)'] = pd.to_numeric(data['Engine Size (L)'], errors="coerce")
print(data['Engine Size (L)'])

# d)
print('Broj vozila marke Audi:')
print(len(data.loc[data['Make'] == 'Audi']))
#print(data.loc[data['Make'] == 'Audi', data['Cylinders' == 4]].mean(data['CO2 Emissions (g/km)']))

# e)
grouped = data.groupby('Cylinders')
print(grouped.mean('CO2 Emissions (g/km)'))

# f)
print(data.groupby('Fuel Type').mean('Fuel Consumption City (L/100km)'))
print(data.groupby('Fuel Type').median('Fuel Consumption City (L/100km)'))

# g)
print(data.loc[data['Cylinders'] == 4].sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1))

# h)

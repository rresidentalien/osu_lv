import pandas as pd

# a)
print('zadatak 1. a)')
data = pd.read_csv('data_C02_emission.csv')
print(len(data))
print(data.info())
data.dropna(axis=0)
data.drop_duplicates()
data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']] = data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].astype('category')
print(len(data))

# b)
print('\nzadatak 1. b)')
print('Najveca potrosnja:', data[['Make','Model','Fuel Consumption City (L/100km)']].sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3))
print('Najmanja potrosnja:', data[['Make','Model','Fuel Consumption City (L/100km)']].sort_values(by='Fuel Consumption City (L/100km)', ascending=True).head(3))

# c)
print('\nzadatak 1. c)')
data['Engine Size (L)'] = data['Engine Size (L)'].astype(str).str.replace(",", ".")
data['Engine Size (L)'] = pd.to_numeric(data['Engine Size (L)'], errors="coerce")
print('Broj vozila s velicinom motora izmedju 2.5 i 3.5:', len(data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]))
print('Njihova prosjecna CO2 emisija:', data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]['CO2 Emissions (g/km)'].mean())

# d)
print('\nzadatak 1. d)')
print('Broj vozila marke Audi:', len(data.loc[data['Make'] == 'Audi']))
print('Prosjecna emisija CO2 Audija s 4 cilindra:', data[(data['Make'] == 'Audi') & (data['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean())

# e)
print('\nzadatak 1. e)')
print('Broj vozila po broju cilindara:', data['Cylinders'].value_counts().sort_index())
grouped = data.groupby('Cylinders')
print('Prosjecna emisija po broju cilindara:', grouped['CO2 Emissions (g/km)'].mean())

# f)
print('\nzadatak 1. f)')
print('Prosjecna gradska potrosnja dizel automobila:', data[data['Fuel Type'] == 'D']['CO2 Emissions (g/km)'].mean())
print('Prosjecna gradska potrosnja benzin automobila:', data[data['Fuel Type'] == 'X']['CO2 Emissions (g/km)'].mean())
print('Medijalna gradska potrosnja dizel automobila:', data[data['Fuel Type'] == 'D']['CO2 Emissions (g/km)'].median())
print('Medijalna gradska potrosnja benzin automobila:', data[data['Fuel Type'] == 'X']['CO2 Emissions (g/km)'].median())

# g)
print('\nzadatak 1. g)')
print('Vozilo s dizel motorom i 4 cilindra koje ima najvecu gradsku potrosnju:', data.loc[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')].sort_values(by='Fuel Consumption City (L/100km)', ascending=False)[['Make', 'Model']].head(1))

# h)
print('\nzadatak 1. h)')
print('Broj vozila s rucnim tipom mjenjaca:', len(data[data['Transmission'].str[0] == 'M']))

# i)
print('\nzadatak 1. i)')
print('Korelacija izmedju numerickih velicina:', data.corr(numeric_only=True))

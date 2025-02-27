import pandas as pd

# Create a simple DataFrame
data = {
    'name': ['John', 'Jane', 'Jim'],
    'age': [15, 16, 14],
    'score': [89, 95, 92]
}
df = pd.DataFrame(data)

print("______")
print(df)

# print(df['name'])
print(df.name)

print("17:", df.loc[1])



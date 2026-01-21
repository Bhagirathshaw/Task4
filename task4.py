import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("adult.csv")

df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

num_cols = [
    'age','fnlwgt','education-num',
    'capital-gain','capital-loss','hours-per-week'
]

cat_cols = [
    'workclass','education','marital-status',
    'occupation','relationship','race',
    'sex','native-country'
]

le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

df = pd.get_dummies(df, columns=cat_cols)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv("adult_processed.csv", index=False)

print(df.head())

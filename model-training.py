import plotly.express as px
import pandas as pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# load data
df_iris = px.data.iris()
X = df_iris.drop(columns=['species', 'species_id'])
y = df_iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=70)
lr = LogisticRegression()
lr.fit(X_train, y_train)

with open("saved-iris-model-2.pkl", "wb") as file:
    pickle.dump(lr, file)

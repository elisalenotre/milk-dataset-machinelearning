import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("milknew.csv")

features = ["pH","Temprature","Taste","Odor","Fat" ,"Turbidity","Colour","Grade"] #  On récupère les colonnes
#features = ["Pclass", "Age"] #  On récupère les colonnes
y = train_data["Grade"] #  la colonne à prédire
x = train_data[features] # on s'entraine sur X

X = pd.get_dummies(x, drop_first=True)  # transformer le texte en chiffres
X["Odor"].fillna(X["Odor"].mean(), inplace=True)# remplir par la moyenne d'age
X["Fat"].fillna(X["Fat"].mean(), inplace=True)

for i in range(100, 130, 2) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)   #configuration random forest , nombre de branches, profondeur ...

    model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)

    model.fit(X_train, y_train)  # notre modele random forest s'entraine ( machine learning )

    predictions = model.predict(X_test) # notre modèle va essayer de prédire  le taux de survie

    accuracy = accuracy_score(y_test, predictions) #resultat

    print(f"L'accuracy du modèle sur l'ensemble de test est de {accuracy * 100:.1f}%")
    print("---------------------------------------------------------------------")




sns.pairplot(train_data, hue="Fat")

plt.show()
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
train_data = pd.read_csv("milknew_enriched.csv")

# # Ajouter une nouvelle colonne "Race de vache" avec des valeurs aléatoires
# races = ["Holstein", "Jersey", "Normande", "Montbeliarde", "Charolaise"]
# train_data["CowRace"] = np.random.choice(races, size=len(train_data))
# train_data.to_csv("milknew_enriched.csv", index=False) #sauvegarde le nv csv

# print("Le dataset a été enrichi et sauvegardé sous 'milknew_enriched.csv'.")


# Définition des features et de la cible
features = ["pH","Temprature","Taste","Odor","Fat","Turbidity","Colour","Grade"] 
y = train_data["Grade"] 
x = train_data[features] 

X = pd.get_dummies(x, drop_first=True)  
X["Odor"].fillna(X["Odor"].mean(), inplace=True)
X["Fat"].fillna(X["Fat"].mean(), inplace=True)

# Boucle d'entraînement du modèle
for i in range(100, 130, 2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

    model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"L'accuracy du modèle sur l'ensemble de test est de {accuracy * 100:.1f}%")
    print("---------------------------------------------------------------------")

# Affichage des données avec Seaborn
sns.pairplot(train_data, hue="Fat")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
train_data = pd.read_csv("milknew_enriched.csv")


# # Ajouter une nouvelle colonne "CowRace" avec des valeurs aléatoires
# races = ["Holstein", "Jersey", "Normande", "Montbeliarde", "Charolaise"]
# train_data["CowRace"] = np.random.choice(races, size=len(train_data)) #sélectionner aléatoirement des éléments de la liste racesx
# train_data.to_csv("milknew_enriched.csv", index=False) #sauvegarde le nv csv

# print("Le dataset a été enrichi et sauvegardé sous 'milknew_enriched.csv'.")


# # Ajouter une nouvelle colonne "Age" avec des valeurs 
# train_data["CowAgeBelowFive"] = train_data["Fat"].apply(lambda x: 1 if x == 1 else 0)
# # Sauvegarder le fichier enrichi
# train_data.to_csv("milknew_enriched2.csv", index=False)

# print("Le dataset a été enrichi avec la colonne 'Age de la vache', puis sauvegardé sous 'milknew_enriched2.csv'.")


# Définition des features et de la cible
# features = ["Temprature","Taste","pH","Fat","Turbidity","Colour"] 99.1%
# features = ["Temprature","pH","Fat"] 91.3%
# features = ["Odor","pH"] 89.4%*
features = ["Turbidity","Fat"] 
y = train_data["Grade"] 
x = train_data[features] 

X = pd.get_dummies(x, drop_first=True)  
X["Turbidity"].fillna(X["Turbidity"].mean(), inplace=True)
# X["pH"].fillna(X["pH"].mean(), inplace=True)
X["Fat"].fillna(X["Fat"].mean(), inplace=True)


# Boucle d'entraînement du modèle
for i in range(1, 20, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

    model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"L'accuracy du modèle Random Forest Classifier sur l'ensemble de test est de {accuracy * 100:.1f}%")
    print("---------------------------------------------------------------------")


sns.pairplot(train_data, hue="Grade")
plt.show()

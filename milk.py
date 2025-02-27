import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
train_data = pd.read_csv("milknew_enriched.csv")

#SCRIPTS D'ENRICHISSEMENT DU DATASET

# # Ajouter une nouvelle colonne "Race de vache" avec des valeurs aléatoires :
#races = ["Holstein", "Jersey", "Normande", "Montbeliarde", "Charolaise"]
#train_data["CowRace"] = np.random.choice(races, size=len(train_data))
#train_data.to_csv("Datasets/old_versions/milknew_enriched.csv", index=False) #sauvegarde le nouveau csv

# # script d'jout de "CowAgeBelowFive" : quand la vache a moins de 5 ans (donc 1), son lait est plus gras
#train_data["CowAgeBelowFive"] = train_data["Fat"].apply(lambda x: 1 if x == 1 else 0)
#train_data.to_csv("Datastets/milknew_enriched2.csv", index=False)


# Définition des features et de la cible
features = ["Taste","Colour"] #faire avec Fat et Colour pour moins d'accuracy
y = train_data["Grade"] 
x = train_data[features] 

X = pd.get_dummies(x, drop_first=True)  
X["Taste"].fillna(X["Taste"].mean(), inplace=True)
X["Colour"].fillna(X["Colour"].mean(), inplace=True)
#X["Odor"].fillna(X["Odor"].mean(), inplace=True)
#X["Turbidity"].fillna(X["Turbidity"].mean(), inplace=True)

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

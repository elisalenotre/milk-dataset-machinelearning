import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
train_data = pd.read_csv("Datasets/milknew_enriched2.csv")

# Définition des features et de la cible
features = ["pH","Temprature","Taste","Odor","Fat","Turbidity","Colour"] 
y = train_data["Grade"] 
x = train_data[features] 

X = pd.get_dummies(x, drop_first=True)  

# Boucle d'entraînement du modèle
for i in range(42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"L'accuracy du modèle Gradient Boosting sur l'ensemble de test est de {accuracy * 100:.1f}%")
    print("---------------------------------------------------------------------")

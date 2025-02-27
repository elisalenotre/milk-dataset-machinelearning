import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# Charger les données
train_data = pd.read_csv("Datasets/milknew_enriched2.csv")

# Définition des features et de la cible
features = ["pH","Temprature","Taste","Odor","Fat","Turbidity","Colour"] 
y = train_data["Grade"] 
x = train_data[features] 

X = pd.get_dummies(x, drop_first=True)  

# Liste des modèles à tester
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear'),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Validation croisée pour chaque modèle
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Accuracy moyenne pour le modèle {name}: {scores.mean() * 100:.1f}%")
    print("---------------------------------------------------------------------")


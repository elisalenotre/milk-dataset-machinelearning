import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train_data = pd.read_csv("Datasets/old_versions/milknew.csv")

# Séparer les features et la target
X = train_data.drop(columns=['Grade'])
y = train_data['Grade']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de forêt aléatoire
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Obtenir les importances des features
feature_importances = clf.feature_importances_

# Créer un DataFrame pour afficher les importances des features
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Trier les features par importance décroissante
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Afficher la feature la plus importante pour trouver le Grade "high"
print("Feature la plus importante pour trouver le Grade 'high':")
print(feature_importances_df.head(1))

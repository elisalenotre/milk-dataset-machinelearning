
import pandas as pd


train_data = pd.read_csv("Datasets/milknew_enriched2.csv")


total_data = len(train_data)

# pourcentage de milk high
data = train_data.loc[train_data.Grade == 'high']
data_rate = len(data) / total_data * 100
print("high: {:.2f}%".format(data_rate))

# pourcentage de milk medium
data = train_data.loc[train_data.Grade == 'medium']
data_rate = len(data) / total_data * 100
print("medium: {:.2f}%".format(data_rate))

# pourcentage de milk low
data = train_data.loc[train_data.Grade == 'low']
data_rate = len(data) / total_data * 100
print("low: {:.2f}%".format(data_rate))

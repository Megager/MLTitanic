import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


def is_has_cabin(row):
    cabin = row["Cabin"]
    if pd.isna(cabin):
        return 0
    elif cabin.startswith("A"):
        return 1
    elif cabin.startswith("B"):
        return 2
    elif cabin.startswith("C"):
        return 3
    elif cabin.startswith("D"):
        return 4
    elif cabin.startswith("E"):
        return 5
    else:
        return 6


def process_data(dataset):

    # Replace undefined age
    dataset["Age"] = dataset.Age.fillna(dataset.Age.mean())

    # Feature engineering
    dataset["Embarked"].fillna("S", inplace=True)
    dataset["IsFamily"] = dataset["Parch"] + dataset["SibSp"]
    dataset["IsFamily"].loc[dataset["IsFamily"] > 0] = 1
    dataset["Family_Size"] = dataset["Parch"] + dataset["SibSp"]
    dataset["CabinType"] = dataset.apply(lambda row: is_has_cabin(row), axis=1)
    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)
    dataset["FareBin"] = pd.qcut(dataset["Fare"], 5)
    label = LabelEncoder()
    dataset["FareBin"] = label.fit_transform(dataset["FareBin"])
    dataset["AgeBin"] = pd.qcut(dataset["Age"], 4)
    label = LabelEncoder()
    dataset["AgeBin"] = label.fit_transform(dataset["AgeBin"])

    # Drop useless features
    dataset.drop(["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Fare", "Age"],
                 axis=1, inplace=True)

    # Create dummies
    dataset["Sex"] = pd.get_dummies(dataset["Sex"])
    dataset["Embarked"] = pd.get_dummies(dataset["Embarked"])

    return dataset


def group_by_family_survival(data):
    data["Last_Name"] = data["Name"].apply(lambda x: str.split(x, ",")[0])
    data["Fare"].fillna(data["Fare"].mean(), inplace=True)
    default_value = 0.5
    data["Family_Survival"] = default_value

    # group families by last name and fare for creating new feature which correspond to survival of all family
    for _, grouped_by_last_name in data.groupby(["Last_Name", "Fare"]):
        if len(grouped_by_last_name) != 1:
            for index, row in grouped_by_last_name.iterrows():
                survived_max = grouped_by_last_name.drop(index)["Survived"].max()
                survived_min = grouped_by_last_name.drop(index)["Survived"].min()
                passenger_id = row["PassengerId"]
                if survived_max == 1.0:
                    data.loc[data["PassengerId"] == passenger_id, "Family_Survival"] = 1
                elif survived_min == 0.0:
                    data.loc[data["PassengerId"] == passenger_id, "Family_Survival"] = 0

    # group families by ticket for creating new feature which correspond to survival of all family
    for _, group_by_ticket in data.groupby("Ticket"):
        if len(group_by_ticket) != 1:
            for index, row in group_by_ticket.iterrows():
                if (row["Family_Survival"] == 0) | (row["Family_Survival"] == 0.5):
                    survived_max = group_by_ticket.drop(index)["Survived"].max()
                    survived_min = group_by_ticket.drop(index)["Survived"].min()
                    passenger_id = row["PassengerId"]
                    if survived_max == 1.0:
                        data.loc[data["PassengerId"] == passenger_id, "Family_Survival"] = 1
                    elif survived_min == 0.0:
                        data.loc[data["PassengerId"] == passenger_id, "Family_Survival"] = 0

    return data["Family_Survival"]


# process train data
data_train = pd.read_csv("train.csv")
data_train = process_data(data_train)

# set Family Survived feature
all_family_survived = group_by_family_survival(pd.read_csv("train.csv").append(pd.read_csv("test.csv")))
data_train["FamilySurvived"] = all_family_survived[:891]

# create model and feature scaler
model = RandomForestClassifier(100)
standard_scaler = StandardScaler()

# crate train and cross-validation sets for testing
X_train = data_train.sample(frac=0.8)
X_test = data_train.drop(X_train.index)

y_test = X_test["Survived"]
X_test.drop("Survived", axis=1, inplace=True)
X_test = standard_scaler.fit_transform(X_test)

y_train = X_train["Survived"]
X_train.drop("Survived", axis=1, inplace=True)
X_train = standard_scaler.fit_transform(X_train)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# create sets for submit predictions
Y_train = data_train["Survived"]
data_train.drop("Survived", axis=1, inplace=True)
X_train = standard_scaler.fit_transform(data_train)

model.fit(X_train, Y_train)

data_test = pd.read_csv("test.csv")

ids = pd.DataFrame(data_test["PassengerId"])

data_test = process_data(data_test)
data_test["FamilySurvived"] = all_family_survived[891:]
data_test = standard_scaler.fit_transform(data_test)

ids["Survived"] = model.predict(data_test)

ids.to_csv("result.csv", index=False)
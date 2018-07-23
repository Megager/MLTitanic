
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# Read input data
titanic_passengers_train = pd.read_csv('train.csv')
titanic_passengers_train.reindex(np.random.permutation(titanic_passengers_train.index))

# Feature engineering
titanic_passengers_train["Embarked"].fillna("S", inplace=True)
titanic_passengers_train["IsFamily"] = titanic_passengers_train["Parch"] + titanic_passengers_train["SibSp"]
titanic_passengers_train["IsFamily"].loc[titanic_passengers_train["IsFamily"] > 0] = 1

# Drop useless features
titanic_passengers_train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Create dummies
titanic_passengers_train["Embarked"] = pd.get_dummies(titanic_passengers_train["Embarked"])
titanic_passengers_train["Sex"] = pd.get_dummies(titanic_passengers_train["Sex"])

# Replace undefined age
titanic_passengers_train["Age"] = titanic_passengers_train.Age.fillna(titanic_passengers_train.Age.mean())
print(titanic_passengers_train.head())

train = titanic_passengers_train.sample(frac=0.8)
cross_validation = titanic_passengers_train.drop(train.index)

Y_train = train["Survived"]
Y_cross_validation = cross_validation["Survived"]
train.drop("Survived", axis=1, inplace=True)
cross_validation.drop("Survived", axis=1, inplace=True)

X_train = train.values
X_cross_validation = cross_validation.values

# Train model
model = RandomForestClassifier(1000)
model.fit(X_train, Y_train)
print(model.score(X_cross_validation, Y_cross_validation))
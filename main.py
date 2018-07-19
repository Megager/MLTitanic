
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Read input data
titanic_passengers_train = pd.read_csv('train.csv')
titanic_passengers_train.reindex(
    np.random.permutation(titanic_passengers_train.index))

# Drop useless features
titanic_passengers_train = titanic_passengers_train.drop("PassengerId", 1)
titanic_passengers_train = titanic_passengers_train.drop("Name", 1)
titanic_passengers_train = titanic_passengers_train.drop("Ticket", 1)
titanic_passengers_train = titanic_passengers_train.drop("Cabin", 1)

# Create dummies
titanic_passengers_train["Embarked"] = pd.get_dummies(titanic_passengers_train["Embarked"])
titanic_passengers_train["Sex"] = pd.get_dummies(titanic_passengers_train["Sex"])

# Replace undefined age
titanic_passengers_train["Age"] = np.where(np.isnan(titanic_passengers_train.Age), 99 , titanic_passengers_train.Age)

Y = titanic_passengers_train["Survived"]
titanic_passengers_train = titanic_passengers_train.drop("Survived", 1)

X = titanic_passengers_train.values

# Train model
model = RandomForestClassifier()
model.fit(titanic_passengers_train, Y)
print(model.score(X, Y))
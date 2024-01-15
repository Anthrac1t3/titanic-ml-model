# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# Set features to train on
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

from sklearn.impute import SimpleImputer

# Clean up data
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data["Age"] = imp.fit_transform(train_data[["Age"]]).ravel()
test_data["Age"] = imp.fit_transform(test_data[["Age"]]).ravel()
test_data["Fare"] = imp.fit_transform(test_data[["Fare"]]).ravel()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

#print(train_data.head())
y = train_data["Survived"]
X = pd.get_dummies(train_data[features])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)

# Specify the possible parameters for the random grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#print(random_grid)

# Create the ml model
model = RandomForestClassifier(random_state=1)
# Hunt for the best hyperparameters
model_random = RandomizedSearchCV(estimator=model,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=3,
                                  verbose=2,
                                  random_state=1,
                                  n_jobs=-1)
# Display the best parameters
print(model_random.get_params())

# Fit the model to the data
model_random.fit(X_train, y_train)
predictions = model_random.predict(X_val)
print((predictions == y_val).sum()/len(predictions)*100)

#model_random.fit(X, y)
#X_test = pd.get_dummies(test_data[features])
#predictions = model_random.predict(X_test)

#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('submission.csv', index=False)
#print("The submission is ready!")
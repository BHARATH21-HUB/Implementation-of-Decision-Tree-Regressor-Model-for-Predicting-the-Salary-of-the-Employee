# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

Program:

## Program:
```py
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

uploaded = files.upload()
file = list(uploaded.keys())[0]

data = pd.read_csv(file)

print("Data shape:", data.shape)
display(data.head())

X = data.drop("Salary", axis=1)
y = data["Salary"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()

```

## Output:

<img width="1639" height="1090" alt="image" src="https://github.com/user-attachments/assets/09957c73-bb87-4a1d-9add-4a5972b80c67" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.


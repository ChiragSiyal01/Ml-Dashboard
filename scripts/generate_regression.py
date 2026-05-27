import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# CHECK
if len(sys.argv) < 2:
    print("CSV path not provided")
    sys.exit(1)

csv_path = sys.argv[1]

# LOAD
data = pd.read_csv(csv_path)

# AUTO SPLIT
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# choose first feature for plotting
x_col = X.columns[0]

# create plotting range
X_range = np.linspace(X[x_col].min(), X[x_col].max(), 200)

# dynamic X_plot
X_plot = pd.DataFrame()
for col in X.columns:
    if col == x_col:
        X_plot[col] = X_range
    else:
        X_plot[col] = X[col].mean()

# OUTPUT DIR
output_dir = "static/graphs/regression"
os.makedirs(output_dir, exist_ok=True)

# plot function
def save_plot(y_pred, title, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(X[x_col], y, alpha=0.3)
    plt.plot(X_range, y_pred, color="red")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("Target")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# MODELS

# Linear
lr = LinearRegression()
lr.fit(X, y)
save_plot(lr.predict(X_plot), "Linear Regression", "linear.png")

# Polynomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_plot_poly = poly.transform(X_plot)

pr = LinearRegression()
pr.fit(X_poly, y)
save_plot(pr.predict(X_plot_poly), "Polynomial Regression", "polynomial.png")

# SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

svr = SVR()
svr.fit(X_scaled, y_scaled)

y_svr = scaler_y.inverse_transform(
    svr.predict(scaler_X.transform(X_plot)).reshape(-1, 1)
).ravel()

save_plot(y_svr, "SVR", "svr.png")

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X, y)
save_plot(dt.predict(X_plot), "Decision Tree", "decision_tree.png")

# Random Forest
rf = RandomForestRegressor()
rf.fit(X, y)
save_plot(rf.predict(X_plot), "Random Forest", "random_forest.png")

print("Regression graphs generated")
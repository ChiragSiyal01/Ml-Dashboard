import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

csv_path = sys.argv[1]
data = pd.read_csv(csv_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

out = os.path.join(ROOT, "static", "graphs", "regression")
os.makedirs(out, exist_ok=True)

# plot data
X_plot = pd.DataFrame()
for col in X.columns:
    if col == X.columns[0]:
        X_plot[col] = np.linspace(X[col].min(), X[col].max(), 200)
    else:
        X_plot[col] = X[col].mean()

def save_plot(pred, name):
    plt.figure()
    plt.scatter(X.iloc[:,0], y)
    plt.plot(X_plot.iloc[:,0], pred, color="red")
    plt.title(name)

    path = os.path.join(out, f"{name}.png")
    plt.savefig(path)
    plt.close()

models = {
    "Linear": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(),
    "Random_Forest": RandomForestRegressor(),
}

metrics = {}

for name, model in models.items():
    model.fit(X, y)
    pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    metrics[name] = (rmse, r2)

    pred_plot = model.predict(X_plot)
    save_plot(pred_plot, name)

# Best model
best = min(metrics, key=lambda x: metrics[x][0])

# Save metrics
with open(os.path.join(out, "metrics.txt"), "w") as f:
    for m in metrics:
       f.write(f"{m} - RMSE:{metrics[m][0]:.2f}, R2:{metrics[m][1]:.2f}\n")

    f.write(f"\n BEST MODEL: {best}")

import matplotlib.pyplot as plt

#  Create bar chart for RMSE
names = list(metrics.keys())
rmse_values = [metrics[m][0] for m in names]

plt.figure(figsize=(8,5))
plt.bar(names, rmse_values)
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.title("Model Comparison (Lower is Better)")
plt.xticks(rotation=30)

bar_path = os.path.join(out, "comparison_bar.png")
plt.savefig(bar_path)
plt.close()

print(" Bar chart saved:", bar_path)
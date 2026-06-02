import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if len(sys.argv) < 2:
    print(" Please provide dataset path")
    sys.exit(1)

csv_path = sys.argv[1]
print(" CSV:", csv_path)

#  Load data
data = pd.read_csv(csv_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#  Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#  Output folder
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

out = os.path.join(ROOT, "static", "graphs", "classification")
os.makedirs(out, exist_ok=True)

#  Models
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier()
}

metrics = {}

#  Train models + save confusion matrix
for name, model in models.items():
    print(" Training:", name)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    metrics[name] = acc

    plt.figure(figsize=(5,5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
    plt.title(name)

    path = os.path.join(out, f"{name}.png")
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

#  Best model (highest accuracy)
best = max(metrics, key=metrics.get)

#  Save metrics file (for UI)
metrics_path = os.path.join(out, "metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:
    for m in metrics:
        f.write(f"{m} - Accuracy:{metrics[m]:.2f}\n")
    f.write(f"\nBEST MODEL: {best}")

print(" Metrics saved:", metrics_path)

#  BAR CHART (NEW )
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(8,5))
plt.bar(names, values)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Classification Model Comparison")
plt.xticks(rotation=30)

bar_path = os.path.join(out, "classification_bar.png")
plt.savefig(bar_path)
plt.close()

print("Bar chart saved:", bar_path)
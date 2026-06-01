import sys, os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

csv_path = sys.argv[1]
data = pd.read_csv(csv_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

out = os.path.join(ROOT, "static", "graphs", "classification")
os.makedirs(out, exist_ok=True)

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier()
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    metrics[name] = acc

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name)

    plt.savefig(os.path.join(out, f"{name}.png"))
    plt.close()

best = max(metrics, key=metrics.get)

with open(os.path.join(out, "metrics.txt"), "w") as f:
    for m in metrics:
        f.write(f"{m} → Accuracy:{metrics[m]:.2f}\n")
    f.write(f"\n BEST MODEL: {best}")
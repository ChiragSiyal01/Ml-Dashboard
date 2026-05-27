import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if len(sys.argv) < 2:
    print("CSV path not provided")
    sys.exit(1)

csv_path = sys.argv[1]

data = pd.read_csv(csv_path)

# convert categorical columns
data = pd.get_dummies(data)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

output_dir = "static/graphs/classification"
os.makedirs(output_dir, exist_ok=True)

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(name)
    plt.savefig(f"{output_dir}/{name}.png")
    plt.close()

print("Classification graphs generated")
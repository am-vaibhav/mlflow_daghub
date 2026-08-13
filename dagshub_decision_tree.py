import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='vaibhav.vaibhav.rai009', repo_name='mlflow_daghub', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/vaibhav.vaibhav.rai009/mlflow_daghub.mlflow")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

mlflow.set_experiment("iris-classification")

n_estimators = 100
max_depth = 5

with mlflow.start_run():
    model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=23
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, name="decision_tree_model")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=[10, 5])
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)
    mlflow.set_tag("author","ankit")
    mlflow.set_tag("model","decision_tree")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
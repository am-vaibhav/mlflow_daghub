# MLflow Tutorial - Iris Classification

## What is MLflow?

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four main components:

1. **MLflow Tracking** - Log parameters, metrics, and artifacts from experiments
2. **MLflow Projects** - Package ML code in a reusable format
3. **MLflow Models** - Deploy models from any ML library
4. **MLflow Model Registry** - Centralized model store for versioning and stage transitions

---

## Project Structure

```
mlflow/
├── docs/
│   └── tutorial.md              # This tutorial
├── mlflow_locally.py            # Local experiment (SQLite backend)
├── mlflow_aws.py                # Remote experiment (EC2 MLflow server)
├── autolog.py                   # Autologging demo (no manual log calls)
├── dagshub_decision_tree.py     # DagsHub remote tracking (Decision Tree)
├── train.py                     # DagsHub remote tracking (Random Forest)
├── ec2_setup.txt                # Steps to deploy MLflow server on AWS EC2
├── mlflow.db                    # Local SQLite backend store
├── mlruns/                      # Local artifact store
└── confusion_matrix.png         # Logged artifact example
```

---

## 1. Installation

```bash
pip install mlflow scikit-learn matplotlib seaborn
```

---

## 2. Running MLflow Locally

**File:** `mlflow_locally.py`

This script trains a Random Forest on the Iris dataset and logs everything to a local SQLite database.

### Key Concepts

#### Setting the Tracking URI

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

- Tells MLflow to store **metadata** (params, metrics, tags, run info) in a local SQLite file.
- `///` = relative path (current directory). `////` = absolute path.
- Without this, MLflow defaults to a `./mlruns` folder for everything.

#### Creating an Experiment

```python
mlflow.set_experiment("iris-classification")
```

- Groups related runs under one experiment name.
- Creates the experiment if it doesn't exist.

#### Starting a Run

```python
with mlflow.start_run():
    # training and logging code here
```

- Opens a new run inside the experiment.
- The `with` block auto-ends the run when done.

#### Logging Parameters

```python
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 5)
```

- Records hyperparameters. These are **input** values to your experiment.

#### Logging Metrics

```python
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)
```

- Records evaluation results. These are **output** values from your experiment.

#### Logging Models

```python
mlflow.sklearn.log_model(model, name="random_forest_model")
```

- Saves the trained model as an artifact.
- MLflow supports sklearn, pytorch, tensorflow, xgboost, and more.

#### Logging Artifacts

```python
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact(__file__)
```

- Saves any file (plots, data files, scripts) alongside the run.
- `__file__` logs the script itself for reproducibility.

#### Setting Tags

```python
mlflow.set_tag("author", "vaibhav")
mlflow.set_tag("model", "random_forest")
```

- Adds searchable metadata to the run (who ran it, model type, etc.).

---

## 3. Viewing Results in MLflow UI

### Local (SQLite backend)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in your browser.

### Default (mlruns folder)

```bash
mlflow ui
```

---

## 4. Remote Tracking with DagsHub

**Files:** `train.py`, `dagshub_decision_tree.py`

DagsHub provides free hosted MLflow tracking servers.

### Setup

```python
import dagshub
dagshub.init(repo_owner='your-username', repo_name='your-repo', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/your-username/your-repo.mlflow")
```

### Install

```bash
pip install dagshub
```

This sends all params, metrics, and artifacts to DagsHub's remote MLflow server instead of local storage.

---

## 5. Autologging

**File:** `autolog.py`

Instead of manually calling `log_param()`, `log_metric()`, `log_model()`, MLflow can auto-log everything.

```python
mlflow.autolog()
```

- Automatically logs parameters, metrics, and the model for supported libraries (sklearn, xgboost, pytorch, etc.)
- Must be called **before** `model.fit()`
- No need for manual `log_param()` or `log_metric()` calls
- Still works inside `with mlflow.start_run():`

---

## 6. Remote Tracking with AWS EC2

**Files:** `mlflow_aws.py`, `ec2_setup.txt`

### Client-Side Code

```python
mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>")
mlflow.set_experiment("iris-classification")
```

Only the tracking URI changes. All logging code stays the same.

### Server Setup (see `ec2_setup.txt` for full steps)

1. Launch EC2 instance (Ubuntu, t3.small minimum)
2. Install Python, pip, mlflow, boto3
3. Configure AWS CLI for S3 artifact storage
4. Add swap space (needed for t3.small with 2GB RAM)
5. Run the MLflow server:

```bash
sudo mlflow server \
    --backend-store-uri sqlite:///home/ubuntu/mlflow/mlflow.db \
    --artifacts-destination s3://your-bucket-name \
    --host 0.0.0.0 \
    --port 80 \
    --workers 1 \
    --allowed-hosts '*' \
    --cors-allowed-origins 'http://<EC2_PUBLIC_IP>'
```

6. Access the UI at `http://<EC2_PUBLIC_IP>`

### Key Server Flags

| Flag | Why It's Needed |
|---|---|
| `--allowed-hosts '*'` | MLflow blocks requests from public IPs by default (DNS rebinding protection). Without this, browser requests get 403. |
| `--cors-allowed-origins` | Browser AJAX requests include an Origin header. Without this, the UI's POST requests (search runs, etc.) get 403. Not needed if using SSH tunnel. |
| `--workers 1` | Reduces memory usage on small instances. Default is 4. |
| `sudo` | Required for port 80 (ports < 1024 need root). Not needed for port 5000+. |

### Gotchas

- **sudo + AWS credentials**: `aws configure` saves creds in `/home/ubuntu/.aws/`, but sudo runs as root and looks in `/root/.aws/`. Copy them: `sudo cp /home/ubuntu/.aws/* /root/.aws/`
- **Swap space**: t3.small (2GB RAM) gets OOM-killed by MLflow's worker processes. Add 2GB swap before starting.
- **ISP blocking**: Some ISPs block non-standard ports (5000). Use port 80 instead, or SSH tunnel: `ssh -L 5000:localhost:5000 ubuntu@<IP>`

### Architecture

```
Client (your laptop)
    │
    ├── Params/Metrics ──> EC2 MLflow Server ──> SQLite (backend store)
    │
    └── Artifacts ──────> EC2 MLflow Server ──> S3 Bucket (artifact store)
```

---

## 7. Backend Store vs Artifact Store

| | Backend Store | Artifact Store |
|---|---|---|
| **Stores** | Params, metrics, tags, run metadata | Models, plots, files |
| **Local default** | `./mlruns` (flat files) | `./mlruns` |
| **With SQLite** | `mlflow.db` file | `./mlruns` (unchanged) |
| **Production** | PostgreSQL / MySQL | S3 / GCS / Azure Blob |
| **Set via** | `set_tracking_uri()` | `--artifacts-destination` on server |

---

## 8. Common MLflow Commands

```bash
# Launch UI locally
mlflow ui

# Launch UI with SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start a tracking server
mlflow server --host 0.0.0.0 --port 5000

# Serve a logged model
mlflow models serve -m "runs:/<RUN_ID>/random_forest_model" --port 1234

# List experiments
mlflow experiments search
```

---

## 9. Workflow Summary

```
1. Write training script
        │
2. Set tracking URI (local SQLite / remote server / DagsHub)
        │
3. Create/set experiment
        │
4. Start a run
        │
5. Train model ──> log_param(), log_metric(), log_model(), log_artifact()
        │
6. View results ──> mlflow ui / DagsHub dashboard / remote server
        │
7. Compare runs, pick best model
        │
8. Register & deploy model (optional)
```
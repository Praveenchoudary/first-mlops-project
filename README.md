---

Build, Train & Deploy a Diabetes Prediction Machine Learning Model (FastAPI + Docker + Kubernetes) | Beginner-Friendly MLOps Project

---

Hello everyone 👋
Welcome to this complete, beginner-friendly MLOps tutorial, where we will walk through building, training, evaluating, containerizing, and deploying a machine learning model using FastAPI, Docker, and Kubernetes.
This guide is ideal for:
✅ DevOps engineers entering ML
✅ Beginners learning MLOps
✅ Anyone exploring real-world ML deployment
✅ Students preparing end-to-end ML projects

Let's dive deep into the world of MLOps 🧠⚙️

---

🔹 Understanding DevOps vs MLOps
Before jumping into coding, let's understand the difference between DevOps & MLOps.
🛠 What is DevOps?
DevOps is the culture of collaboration between:
🧑‍💻 Development
⚙️ Operations

It automates:
Build
Test
Deploy
Monitor

Result?
➡️ Faster releases
 ➡️ Improved reliability

---

🤖 What is MLOps?
MLOps extends DevOps for machine learning systems.
Machine learning lifecycle includes:
📥 Data ingestion
🧹 Data cleaning
🎓 Model training
✅ Model evaluation
📦 Model packaging
☸️ Model deployment
🔁 Continuous retraining

In ML, the challenge is model degradation - new data changes behaviour over time, so retraining is essential.

---

🔹 The Dataset We Are Using 📊
We use a diabetes dataset in CSV format from open datasets available on the internet.
Key columns:
Feature Description Pregnancies Number of pregnancies Glucose Glucose level BloodPressure Blood pressure value BMI Body mass index Age Age of the person Outcome 1 (Diabetic) / 0 (Non-diabetic)

---

🔹 Project Architecture (High-Level)
Here is the simple flow 🚀:
Dataset → Train Model → Save Model.pkl → Build FastAPI API → Dockerize → Deploy to Kubernetes

---

🔹 Project Code
Let's build the project step-by-step.

---

📁 Folder Structure
diabetes-mlops/
│
├── train.py
├── main.py
├── requirements.txt
├── Dockerfile
├── deploy.yaml

---

🔹 Step 1: Model Training Code (train.py)
# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
# Load dataset from a working source (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
print("✅ Columns:", df.columns.tolist())  # Debug print
# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Save
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")

---

🔹 Step 2: FastAPI Application (main.py)
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
app = FastAPI()
model = joblib.load("diabetes_model.pkl")
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int
@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}
@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    return {"diabetic": bool(prediction)}

---

🔹 Step 3: Requirements File (requirements.txt)
fastapi
uvicorn
scikit-learn
pandas
joblib
numpy

---

🔹 Step 4: Test Locally
💻 Create virtual environment:
python3 -m venv .mlops
source .mlops/bin/activate
Install required libraries:
pip install -r requirements.txt
Train the model:
python3 train.py
Run FastAPI server:
uvicorn main:app --reload
Open:
✅ http://127.0.0.1:8000
✅ http://127.0.0.1:8000/docs

---

🔹 Step 5: Dockerize the Project 🐳
Create Dockerfile:
# Dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
Build image:
docker build -t diabetes-model-demo .
Run:
docker run -p 8000:8000 diabetes-model-demo
🔹 Step 6: Push Image to Docker Hub
docker tag diabetes-model-demo username/diabetes-model-demo:v1
docker login
docker push username/diabetes-model-demo:v1

---

🔹 Step 7: Kubernetes Deployment (deploy.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diabetes-api
  labels:
    app: diabetes-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: diabetes-api
  template:
    metadata:
      labels:
        app: diabetes-api
    spec:
      containers:
      - name: diabetes-api
        image: praveen22233/demo-mlop-dia:v1
        ports:
        - containerPort: 8000
        imagePullPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: diabetes-api-service
spec:
  selector:
    app: diabetes-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
~

---

🔹 Step 8: Deploy to Kubernetes ☸️
Deploy:
kubectl apply -f deploy.yaml
Check:
kubectl get pods
kubectl get svc
🔹 Step 9: Access the API via LoadBalancer
Visit:
➡️ http://load-balancer/docs

---

Final Result 🎉
You now have a fully functional:
✅ Machine learning model
✅ FastAPI backend
✅ Docker container
✅ Kubernetes deployment
✅ REST API prediction service

This project covers complete beginner-friendly MLOps workflow, from training to deployment.

---

❤️ Conclusion
MLOps is not difficult when you break it into simple steps.
 If you know Docker, Kubernetes, and basics of Python - you are already 70% there.
In this blog, we covered:
📥 Data ingestion
 🧹 Data preparation
 🎓 Model training
 ✅ Model evaluation
 🐳 Docker containerization
 ☸️ Kubernetes deployment
Connect with me:
👨‍💻 LinkedIn
💻 GitHub

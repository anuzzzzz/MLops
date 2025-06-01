
PROJECT_ID = "dulcet-bastion-452612-v4"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Create a Cloud Storage bucket
# Create a storage bucket to store intermediate artifacts such as datasets.
BUCKET_URI = f"gs://mlops-course-dulcet-bastion-452612-v4-unique"  # @param {type:"string"}

# If your bucket doesn't already exist:
# ! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}

# Initialize Vertex AI SDK for Python
# To get started using Vertex AI, you must have an existing Google Cloud project
# and enable the Vertex AI API (https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

# Import the required libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import pickle
import joblib

# Configure resource names
# Set a name for the following parameters:
# MODEL_ARTIFACT_DIR - Folder directory path to your model artifacts within a Cloud Storage bucket.
# REPOSITORY - Name of the Artifact Repository to create or use.
# IMAGE - Name of the container image that is pushed to the repository.
# MODEL_DISPLAY_NAME - Display name of Vertex AI model resource.
MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"  # @param {type:"string"}
REPOSITORY = "iris-classifier-repo"  # @param {type:"string"}
IMAGE = "iris-classifier-img"  # @param {type:"string"}
MODEL_DISPLAY_NAME = "iris-classifier"  # @param {type:"string"}

# Set the defaults if no names were specified
if MODEL_ARTIFACT_DIR == "[your-artifact-directory]":
    MODEL_ARTIFACT_DIR = "custom-container-prediction-model"
if REPOSITORY == "[your-repository-name]":
    REPOSITORY = "custom-container-prediction"
if IMAGE == "[your-image-name]":
    IMAGE = "sklearn-fastapi-server"
if MODEL_DISPLAY_NAME == "[your-model-display-name]":
    MODEL_DISPLAY_NAME = "sklearn-custom-container"

# Simple Decision Tree model
# Build a Decision Tree model on iris data
data = pd.read_csv('data/iris.csv')
print("First 5 rows of the Iris dataset:")
print(data.head(5))

train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(metrics.accuracy_score(prediction, y_test)))

# Save the trained model
# Creates a directory for artifacts if it doesn't exist
os.makedirs("artifacts", exist_ok=True)
joblib.dump(mod_dt, "artifacts/model.joblib")
print(f"Model saved to artifacts/model.joblib")

# Upload model artifacts and custom code to Cloud Storage
# Before deployment, Vertex AI needs access to model.joblib and preprocessor.pkl in Cloud Storage.
# Note: The notebook mentions preprocessor.pkl but it's not created in the provided code.
# Ensure preprocessor.pkl is handled if needed for your actual deployment.
# Run the following command to upload your model artifact:
!gsutil cp artifacts/model.joblib {BUCKET_URI}/{MODEL_ARTIFACT_DIR}/
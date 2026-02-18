# MLOps Week 01 Assignment

## Objective

Set up a basic ML pipeline in GCP using Vertex AI and Google Cloud Storage.
Train an IRIS classifier, store artifacts in GCS using timestamp folders, and run inference using a separate script.

## Project Files

src/train.py
Trains a RandomForest model using iris dataset from GCS.
Saves model and metrics in a timestamp-based folder inside the GCS bucket.

src/inference.py
Downloads a trained model from GCS (based on timestamp) and runs prediction.

requirements.txt
Contains required Python libraries.

## GCS Structure

Bucket contains:

data/iris.csv → Input dataset

artifacts/<timestamp>/ → Model and metrics from each training run

Each time training is executed, a new timestamp folder is created to avoid overwriting previous models.

How to Run

Train:

python src/train.py


Run inference:

python src/inference.py <timestamp>

## Learnings

Understood how to store and fetch data from GCS

Learned artifact versioning using timestamps

Practiced separating training and inference workflows

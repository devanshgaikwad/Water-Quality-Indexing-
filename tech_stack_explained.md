# 📚 The Tech Stack Explained: Libraries, Frameworks & Core Functions

To truly master this project, you need to understand *what tools* you are using, *why* you are using them, and *how* they execute the core logic of the system. 

This document breaks down every library, framework, and crucial function in your Water Quality Prediction system, explaining them from scratch in simple language.

---

## Part 1: The Libraries & Frameworks

Think of libraries as pre-built toolboxes. Instead of building a hammer from scratch, you just import the "Hammer" library.

### 1. Pandas (`import pandas as pd`)
*   **What it is:** The ultimate spreadsheet tool for Python.
*   **What we use it for:** We use Pandas to load our `water_quality.csv` file into a `DataFrame` (which is just Python's word for a data table). It allows us to easily drop missing values (NaNs), remove duplicate rows, and slice specific columns like `['pH', 'Turbidity']`.

### 2. NumPy (`import numpy as np`)
*   **What it is:** A super-fast math library designed for handling massive grids of numbers (arrays).
*   **What we use it for:** Neural networks don't understand Pandas DataFrames. They only understand raw NumPy arrays. NumPy allows us to instantly reshape our flat 2D data into the 3D box shape `[Samples, Timesteps, Features]` required by the LSTM.

### 3. Scikit-Learn (`import sklearn`)
*   **What it is:** The standard library for traditional Machine Learning tools.
*   **What we use it for:** We don't use it for the AI brain, but we use its helper tools:
    *   `MinMaxScaler`: To shrink our numbers down to a 0.0 – 1.0 scale.
    *   `train_test_split`: To perfectly chop our dataset into 70% Training and 15% Testing chunks.
    *   `metrics`: To calculate our final scores like RMSE, MAE, and Accuracy.

### 4. TensorFlow & Keras (`import tensorflow as tf`)
*   **What it is:** Built by Google, this is the heavy-duty engine that actually builds and trains Deep Learning neural networks. Keras is the "easy-to-use" wrapper built on top of TensorFlow.
*   **What we use it for:** This is the AI. We use Keras to stack our brain layers (`Sequential`, `LSTM`, `Dense`, `Dropout`). We use TensorFlow to do the intense calculus required to train the model over 100 epochs.

### 5. FastAPI (`from fastapi import FastAPI`)
*   **What it is:** A blazing-fast, modern framework for building web APIs.
*   **What we use it for:** It acts as the telephone operator. It sits on port `8000`, listens for incoming internet requests containing sensor data, passes that data to the TensorFlow model, and replies with the WQI prediction. 

### 6. Uvicorn (`uvicorn api.main:app`)
*   **What it is:** An ASGI web server.
*   **What we use it for:** FastAPI is just the *code* for the API. Uvicorn is the actual *engine* that runs that code and keeps the server alive on your computer so it can accept internet traffic.

### 7. Pydantic (`from pydantic import BaseModel`)
*   **What it is:** A strict data validation tool used tightly with FastAPI.
*   **What we use it for:** We define a `BaseModel` that says "pH must be a float between 0 and 14." If a broken sensor sends a pH of 9000, Pydantic acts as a bouncer and rejects the request instantly, protecting the TensorFlow model from crashing.

### 8. Streamlit (`import streamlit as st`)
*   **What it is:** A framework that turns Python scripts into beautiful, interactive web applications instantly.
*   **What we use it for:** We use it to build `dashboard.py`. It creates the sliders, buttons, and visual layout so a human can interact with the AI without needing to write terminal code.

### 9. Matplotlib (`import matplotlib.pyplot as plt`)
*   **What it is:** The oldest and most powerful graphing library in Python.
*   **What we use it for:** We use it in `evaluate.py` to draw the Confusion Matrix and Loss curves. We also use it in Streamlit to draw the real-time green trend line showing the WQI predictions over time.

### 10. Joblib (`import joblib`)
*   **What it is:** A tool for saving Python objects to your hard drive.
*   **What we use it for:** Training a `MinMaxScaler` takes time. We use `joblib.dump()` to save the trained scaler as a `.pkl` file so the API can quickly load it (`joblib.load()`) instantly when the server boots up.

---

## Part 2: The Most Important Functions Explained

Here is exactly how the data flows through your custom functions, from scratch.

### 1. `generate_water_quality_data()` (Inside `generate_data.py`)
*   **How it works:** It uses math formulas (like `np.sin()` for sine waves) to simulate the rising and falling temperatures of a river over a year. It then generates 10,000 rows of fake sensor data based on these waves.
*   **Why it's important:** Without data, an AI cannot learn. Because we don't have a physical river, this function acts as our virtual environment.

### 2. `create_sequences(data, seq_length=24)` (Inside `preprocess.py`)
*   **How it works:** It creates empty lists for `X` (the inputs) and `y` (the answers). It loops through the entire dataset. In step 1, it copies rows 0 through 23 into `X`, and copies row 24 into `y`. In step 2, it copies rows 1 through 24 into `X`, and row 25 into `y`. It repeats this thousands of times.
*   **Why it's important:** LSTMs cannot read single rows. They need "stories." This function converts a flat spreadsheet into thousands of 24-hour stories for the AI to study.

### 3. `preprocess_data()` (Inside `preprocess.py`)
*   **How it works:** This is the orchestrator. It executes a strict checklist:
    1. Fill blanks with Forward/Backward fill.
    2. Split data into Train, Validation, and Test sets (70/15/15).
    3. Fit the `MinMaxScaler` on the Training data *only*.
    4. Scale all sets to numbers between 0 and 1.
    5. Pass the scaled data to `create_sequences()`.
*   **Why it's important:** This is the gatekeeper. It ensures the AI receives perfectly formatted, perfectly scaled 3D data arrays with absolutely zero data leakage.

### 4. `build_model()` (Inside `model.py`)
*   **How it works:** It defines a `Sequential()` pipeline in Keras. It adds an `Input` layer expecting a shape of `(24, 5)`. It stacks an `LSTM` layer, a `Dropout` layer (to prevent memorization), and a final `Dense` layer with a `sigmoid` activation (to force the output between 0 and 1). It compiles the model using the `adam` optimizer.
*   **Why it's important:** This defines the literal physical structure of the "brain" before it learns anything. 

### 5. `model.fit()` (Inside `train.py`)
*   **How it works:** This is a built-in Keras function, but it's the most important action in the project. It feeds `X_train` into the model, compares the output to `y_train`, calculates the error (Loss), and mathematically adjusts the model's internal neurons to reduce that error. It does this over and over for 100 Epochs.
*   **Why it's important:** This is the actual "learning" process. Without it, the model is just an empty box.

### 6. `load_assets()` (Inside `api/main.py` and `app/dashboard.py`)
*   **How it works:** When the server or dashboard starts, this function uses `tf.keras.models.load_model()` to load the `.keras` file from the hard drive into RAM. It uses `joblib.load()` to load the `.pkl` scalers. 
*   **Why it's important:** Loading these heavy files takes several seconds. We load them *once* at startup so that when a user asks for a prediction, the response happens in milliseconds.

### 7. `_predict_from_array()` (Inside `api/main.py`)
*   **How it works:** This is the core engine of real-time prediction. 
    1. It takes a new 24-hour array from a user.
    2. It scales it using `scaler.transform()`.
    3. It wraps it in a 3D box using `np.expand_dims()`.
    4. It asks the brain for a guess using `model.predict()`.
    5. It un-scales the guess back to a normal 0-100 score using `target_scaler.inverse_transform()`.
*   **Why it's important:** This is the Inference Pipeline. It is the culmination of all your hard work, translating a user's raw sensor data into a highly accurate, actionable safety warning instantly.

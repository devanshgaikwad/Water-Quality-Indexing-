# 💧 End-to-End Real-Time Water Quality Prediction (LSTM)

A complete, production-ready machine learning project for **real-time water quality prediction** using **Long Short-Term Memory (LSTM)** neural networks.

The system uses time-series sensor data (pH, Turbidity, Dissolved Oxygen, Temperature, Conductivity) to predict a continuous **Water Quality Index (WQI)** and classify water as **Safe** or **Unsafe**.

---

## 📁 Project Structure

```
water-quality-lstm/
│
├── api/                       # FastAPI backend
│   ├── __init__.py
│   └── main.py                # REST endpoints & prediction logic
│
├── app/                       # Streamlit frontend
│   └── dashboard.py           # Real-time monitoring dashboard
│
├── data/                      # Datasets
│   └── water_quality.csv      # Generated after first training run
│
├── models/                    # Saved model artifacts
│   ├── lstm_wqi_model.keras   # Trained Keras model
│   ├── scaler.pkl             # Fitted MinMaxScaler
│   └── *.png                  # Evaluation plots
│
├── notebooks/                 # Jupyter notebooks
│   └── EDA.ipynb              # Exploratory Data Analysis
│
├── src/                       # Core ML source code
│   ├── __init__.py
│   ├── generate_data.py       # Synthetic data generation
│   ├── preprocess.py          # Scaling, sequencing, train/test split
│   ├── model.py               # Stacked LSTM architecture
│   ├── train.py               # Main training pipeline
│   ├── evaluate.py            # Metrics & plotting
│   ├── simulate.py            # CLI real-time simulation
│   └── logger.py              # Centralized logging
│
├── Dockerfile                 # Container for API
├── docker-compose.yml         # Multi-service deployment
├── make_notebook.py           # EDA notebook generator
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## ✨ Features

| Component | Description |
|-----------|-------------|
| **Data Pipeline** | Missing-value imputation (ffill → bfill → mean), MinMax scaling, sliding-window sequencing |
| **LSTM Model** | 2-layer stacked LSTM with BatchNorm + Dropout, trained with EarlyStopping & ReduceLROnPlateau |
| **Evaluation** | RMSE, MAE, R², Accuracy, Precision, Recall, F1 + training curves, scatter, confusion matrix |
| **FastAPI** | `/predict` (24-step window), `/predict/single` (quick test), `/health`, Swagger docs |
| **Streamlit** | Live dashboard with sensor cards, prediction trend chart, manual input, auto-simulation |
| **CLI Simulation** | Terminal-based real-time prediction with ASCII progress bars |
| **Docker** | Multi-stage Dockerfile + docker-compose for API & dashboard |
| **Logging** | Console + daily rotating log files in `logs/` |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/water-quality-lstm.git
cd water-quality-lstm

# Create virtual environment (Python 3.9+)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

This will automatically:
- Generate 10,000 synthetic hourly sensor records
- Preprocess (impute missing → scale → create sequences)
- Train a stacked LSTM with early stopping
- Save model to `models/lstm_wqi_model.keras`
- Save scaler to `models/scaler.pkl`
- Generate evaluation plots in `models/`

### 3. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 4. Launch the Dashboard

In a new terminal:

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501 to view the real-time monitoring UI.

### 5. CLI Simulation (Optional)

```bash
python src/simulate.py
```

### 6. Generate EDA Notebook (Optional)

```bash
pip install nbformat
python make_notebook.py
```

---

## 🐳 Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Or just the API
docker build -t water-quality-api .
docker run -p 8000:8000 water-quality-api
```

> **Note:** Train the model first so `.keras` and `.pkl` files exist.

---

## 📊 API Usage

### Predict (full sequence)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [{"pH":7.5,"Turbidity":3,"Dissolved_Oxygen":8,"Temperature":20,"Conductivity":300}, ...]}'
```

### Quick test (single reading)

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"pH":7.5,"Turbidity":3,"Dissolved_Oxygen":8,"Temperature":20,"Conductivity":300}'
```

### Response

```json
{
  "predicted_wqi": 82.45,
  "status": "Safe",
  "confidence": "High",
  "latency_ms": 12.34
}
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — LSTM model
- **Pandas, NumPy, Scikit-learn** — data pipeline
- **FastAPI + Uvicorn** — REST API
- **Streamlit** — dashboard
- **Matplotlib, Seaborn** — visualisation
- **Docker** — containerisation

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

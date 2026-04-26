"""
make_notebook.py
----------------
Generates an Exploratory Data Analysis (EDA) Jupyter notebook
for the water quality dataset.

Usage:
    pip install nbformat
    python make_notebook.py
"""

import os
import nbformat as nbf

notebook = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# 📊 Exploratory Data Analysis — Water Quality Dataset\n\n"
        "This notebook explores the synthetic water quality dataset used to train\n"
        "our LSTM model. We look at distributions, correlations, time-series\n"
        "trends, and the Safe/Unsafe class balance."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='darkgrid')\n"
        "%matplotlib inline"
    ),
    nbf.v4.new_markdown_cell("## 1. Load Dataset"),
    nbf.v4.new_code_cell(
        "df = pd.read_csv('../data/water_quality.csv')\n"
        "print(f'Shape: {df.shape}')\n"
        "df.head()"
    ),
    nbf.v4.new_code_cell("df.info()"),
    nbf.v4.new_code_cell("df.describe()"),
    nbf.v4.new_markdown_cell("## 2. Missing Values"),
    nbf.v4.new_code_cell(
        "missing = df.isnull().sum()\n"
        "print(missing[missing > 0])\n"
        "print(f'\\nTotal missing: {df.isnull().sum().sum()}')"
    ),
    nbf.v4.new_markdown_cell("## 3. Feature Distributions"),
    nbf.v4.new_code_cell(
        "features = ['pH', 'Turbidity', 'Dissolved_Oxygen', 'Temperature', 'Conductivity', 'WQI']\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
        "for ax, col in zip(axes.flatten(), features):\n"
        "    df[col].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')\n"
        "    ax.set_title(col)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## 4. Correlation Heatmap"),
    nbf.v4.new_code_cell(
        "corr = df[features].corr()\n"
        "plt.figure(figsize=(8, 6))\n"
        "sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)\n"
        "plt.title('Feature Correlation Matrix')\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## 5. Time-Series Trends"),
    nbf.v4.new_code_cell(
        "df['timestamp'] = pd.to_datetime(df['timestamp'])\n"
        "fig, axes = plt.subplots(3, 2, figsize=(15, 10))\n"
        "for ax, col in zip(axes.flatten(), features):\n"
        "    ax.plot(df['timestamp'][:500], df[col][:500], linewidth=0.8)\n"
        "    ax.set_title(col)\n"
        "    ax.set_xlabel('Time')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## 6. Safe vs Unsafe Distribution"),
    nbf.v4.new_code_cell(
        "counts = df['Status'].value_counts()\n"
        "plt.figure(figsize=(6, 4))\n"
        "counts.plot(kind='bar', color=['#4CAF50', '#F44336'])\n"
        "plt.title('Water Quality Status Distribution')\n"
        "plt.ylabel('Count')\n"
        "plt.xticks(rotation=0)\n"
        "plt.show()\n"
        "print(counts)"
    ),
]

notebook["cells"] = cells

os.makedirs("notebooks", exist_ok=True)
output_path = "notebooks/EDA.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(notebook, f)

print(f"✅ Notebook created: {output_path}")

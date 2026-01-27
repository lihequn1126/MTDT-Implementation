#  MTDT: Multi-scale Temporal Decomposition Transformer

This repository contains the official PyTorch implementation for the paper: 
**"Multi-scale Temporal Decomposition Transformer (MTDT): A Robust Framework for Multi-Regional Load Forecasting"**.

## Project Structure

The repository is organized as follows:

*   **`model/`**: Contains the source code for the MTDT model and baseline models (e.g., LSTM, Transformer, iTransformer). The main training script is located here.
*   **`LoadData/`**: Contains **mock sample data** for electricity load.
*   **`nwpData/`**: Contains **mock sample data** for Numerical Weather Prediction (NWP).
*   **`requirements.txt`**: List of Python dependencies.

## Data Privacy & Reproducibility Note

**Crucial Information for Reviewers:**

Due to strict commercial data privacy agreements, the **real-world industrial datasets** (from Sichuan, China) used in the experiments cannot be publicly released. 

To facilitate code verification and reproducibility, we provide **mock sample datasets** in the `LoadData/` and `nwpData/` folders.
*   These files follow the **exact schema and format** of the real-world data.
*   The values are **randomly generated/anonymized**.
*   **Note:** Consequently, running the code on this mock data will yield training metrics (MAE/RMSE) that differ significantly from the results reported in the paper. This is expected behavior.

## How to Run

### 1. Environment Setup

We recommend using a virtual environment

```bash
# Install dependencies
pip install -r requirements.txt~
```

### 2.Training & Evaluation 

To execute the full pipeline (training and evaluation) in the background, use the `nohup` command. This ensures the process continues running even if the terminal session disconnects. **Run the following command:** 

~~~
nohup python -u ./model/run_all.py > ./model/run.log 2>&1 &
~~~


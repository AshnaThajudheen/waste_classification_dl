# EcoSort AI — Full Updated Application

## Required files
Place your trained `ecosort_best.keras` file in the same folder as `app.py`.

## Run on Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- Sustainability-themed UI with high-contrast prediction card
- Image upload and CNN prediction
- Evaluation dashboard with recorded test metrics
- Prediction history with CSV export and clear option
- Organic and recyclable waste guidance

## Model assumptions
- Input: RGB image resized to 224x224
- The model has its own `Rescaling(1./255)` layer
- Output sigmoid: 0 = Organic (O), 1 = Recyclable (R)
- Reported evaluation: accuracy 86.55%, macro precision 86.38%, macro recall 86.81%, macro F1 86.47%, ROC-AUC 94.08%

## Note
Prediction history is saved to a local CSV file. On some hosting platforms, local files can be reset. The model score is not a calibrated probability or guarantee of correctness.

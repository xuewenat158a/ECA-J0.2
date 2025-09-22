# ECA Random Forest — Predict J0.2 (5 Inputs)

## Run locally
```bash
pip install -r requirements.txt
streamlit run eca_rf_j02_app.py
```

## Expected dataset
Excel file (default: `ECA ML.xlsx`, sheet: `ML database`) with columns:
- Sour region
- pH
- ppH2S(bara)
- K-rate
- Notch Location
- J0.2  (target)

Raw headers are preserved for preview; training uses canonical renamed headers.

# P704 recommendation demo

Run from this folder after installing `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app serves precomputed Approach A clusters and training-only recommendation
lists and compact training-only product statistics from `artifacts/`; it does not
need `ratings.csv` or retrain the model.
The source data contains product IDs only, so the app does not display product
names. For users outside the modeled population, Global Popularity is used;
because no history is stored for those users, their known rated product IDs can
be entered in the optional exclusion field.

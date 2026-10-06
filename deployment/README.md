# P704 recommendation demo

Run from this folder after installing `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app serves saved Approach A recommendation data, training-only product
statistics, and the fitted behavioural scaler/KMeans pipeline from `artifacts/`.
Existing modelled users receive their saved cluster ranking with training-seen
products excluded. New users can enter at least five ratings for unique Product
IDs; their behavioural features are transformed by the saved scaler and assigned
to an initial cluster by the saved KMeans model. The model is not retrained.

Users outside the modelled population who do not use onboarding receive the
Global Popularity fallback; optional known Product IDs can be excluded. The app
does not load `ratings.csv`. Product names are unavailable because the source
contains Product IDs only. Displayed rating statistics are historical and
training-only, not predicted ratings.

Use the sidebar to select a demonstration user or enter a userId manually
(manual input takes precedence), choose Top-N from 1 to 20, and optionally enter
comma-separated product IDs to exclude. Submit with **Get Recommendations**.
The **Recommendations** tab shows saved-user results and **New User** provides
the five-rating onboarding demonstration with suggested Product IDs. Results
include historical training-only statistics; **Model Insights** displays saved
cluster profiles, and **About** summarizes the serving paths.

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

The **Recommendations** tab contains the existing-user controls: select a
demonstration user or enter a User ID (manual input takes precedence), choose
Top-N from 1 to 20, and optionally exclude comma-separated Product IDs. Submit
with **Get Recommendations**.

The **New User** tab starts with five editable sample Product IDs from the
saved Global Popularity ranking; enter ratings and select **Build My
Recommendations**. Additional examples are available in a collapsed browser.
Results show historical training-only statistics. **Model Insights** displays
the saved cluster profiles and population, while **About** summarizes the
serving paths.

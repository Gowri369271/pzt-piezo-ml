# Explainable ML for trapezoidal piezoelectric harvesters

Re-implementation of the ML workflow in:
Islam, Karmaker, Galib, Mohiuddin, Hasan.

"Decoupling the interacting influences
of geometry and coverage in trapezoidal piezoelectric energy harvesters using
explainable machine learning." Energy Conversion and Management: X 31 (2026) 102173.
https://doi.org/10.1016/j.ecmx.2026.102173 (CC BY 4.0)

## Important: about the data
`data/pzt_dataset_extracted.csv` is NOT the authors' dataset. It was digitised
by eye from the paper's figures (70 of the paper's 150 points; a/b = 0.25–4 only).
Power is derived as power density × PZT volume. Readings are approximate
(about ±0.1–0.2 µW/mm³ for power density), so results will not match the paper.

## Run it
Open `notebooks/pzt_ml_pipeline.ipynb` in Google Colab, or:

    pip install -r requirements.txt
    python src/train_paper_pipeline.py data/pzt_dataset_extracted.csv outputs

## Results

Results on the 70-row dataset

Model ranking: this mostly matches the paper. SVR is best for voltage, power and power density, and GBR is best for eigenfrequency. RF is weakest on eigenfrequency.
SHAP: a/b clearly dominates for voltage (3.9 vs 1.7) and eigenfrequency (29 vs 15), as the paper found. Power density ranks a/b above coverage, so I’d treat it as mixed. For power, coverage comes out slightly above a/b, whereas the paper has a/b first.
Instability: with a 14-row test set, repeated CV gives power density R² of 0.935 ± 0.06. Treat the test-set R² values as optimistic.

## Known limitations
Accuracy drops sharply when predicting a/b = 4 or 0.25 (extrapolation).
This project is not affiliated with or endorsed by the paper's authors.

## Licence
Code: MIT. Data: CC BY 4.0, derived from the paper above (attribution required).

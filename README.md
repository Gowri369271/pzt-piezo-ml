# Explainable ML for trapezoidal piezoelectric harvesters

Re-implementation of the ML workflow in:
Islam, Karmaker, Galib, Mohiuddin, Hasan. "Decoupling the interacting influences
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

## Known limitations
Accuracy drops sharply when predicting a/b = 4 or 0.25 (extrapolation).
This project is not affiliated with or endorsed by the paper's authors.

## Licence
Code: MIT. Data: CC BY 4.0, derived from the paper above (attribution required).
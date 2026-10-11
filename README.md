# Intelligent Data Cleaner

A local Streamlit application for multi-file CSV and Excel worksheet preview, quality profiling, independent EDA, relationship analysis, configurable cleaning, verification, and export. Use Python 3.11–3.13 for a compatible Streamlit/PyArrow wheel set.

## Run on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt && streamlit run app.py`.

## Use

Upload multiple CSV, XLSX, or XLS files on **Upload & Dataset Selection**. Every worksheet in a workbook is listed independently. EDA runs on the original by default, and its HTML, JSON, and CSV reports are available without cleaning. Configure missing value handling in **Imputation Methods**, apply cleaning explicitly, review automatic verification, then export results. Original uploads remain unchanged for the current session. Use **Clear all uploaded data** to clear session data. Example fixtures are in `sample_data/`.

The application keeps data in session memory and does not persist it to disk. Export anything needed before a session ends. Workloads are limited by local memory and Streamlit upload limits. Large pairwise/categorical analyses are intentionally bounded by selection/category safeguards. Completeness describes missingness only, not correctness. See About & Methodology in the app.

## Tests

Run `python -m pytest`. The tests exercise CSV parsing, type classification, missing values, statistics, duplicates, imputation, verification, and report serialization.

## Layout

`app.py` provides the multipage workflow; `src/core.py` contains independent processing functions; `sample_data/` contains small fixtures.

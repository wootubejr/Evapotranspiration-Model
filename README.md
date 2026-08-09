# SEBAL-ET / ML Irrigation Classification Pipeline

Scripts for LSE postgraduate dissertation: SEBAL-based evapotranspiration modelling over
Spanish SIGPAC parcels, used as a feature set to classify irrigated vs. rainfed
parcels via machine learning.

Run in the order below. Each stage's output is the next stage's input.

---

## 1. `et_model_v3.py` — SEBAL energy balance / ET modelling engine

**What it does:** implements the full SEBAL chain for a single area of interest
(AOI) and date range — spectral indices, net radiation, soil heat flux, sensible
heat flux via iterative Monin-Obukhov stability correction, and final
ET_inst / EF / ET24h assembly. Source imagery is **Landsat 8/9** (Collection 2,
Level-2, via Microsoft Planetary Computer's STAC catalog, 30 m resolution),
with a Copernicus 30 m DEM for elevation/slope corrections and either ERA5 or a
fixed reference wind speed.

**Entry point:** `run_et_for_aoi(...)`. This file has no `__main__` block or CLI
of its own — it's a function library, called by driver scripts:
- **`main.py`** — a lightweight pilot/smoke-test driver. Loops row-by-row over a
  single geometry file (i.e. .shp or .gpkg), calling `run_et_for_aoi`
  once per tile and exporting a raster per tile via `raster_export()`. Useful for
  testing the engine on a small AOI before committing to a national run.
  Includes a `WIND_SOURCE` toggle (`"era5"` vs `"fixed"`) — flip and rerun for
  the ERA5-vs-fixed-wind design comparison.

---

## 2. `phase3_national.py` — national-scale batch driver

**What it does:** the production run across all 50 Spanish provinces. Loads
parcels from GPKG files, builds a 50 km multi-UTM-zone-aware grid
(`build_national_grid_tiles`), assigns parcels to tiles via a vectorized spatial
join (`assign_parcels_to_tiles`). For each occupied tile it then:
discovers cloud-free Landsat dates, runs `run_et_for_aoi` per tile/date,
computes parcel-level zonal statistics, drops parcels below `MIN_PIXEL_COUNT`
valid pixels, and aggregates across the date trajectory (mean/std per feature)
into the final national parcel-level feature table that Phase 4 consumes.

**Entry point:** directly runnable — `python phase3_national.py`. Has a proper
`if __name__ == "__main__":` block.

**Config to check/set before running:**
| Variable | Purpose |
|---|---|
| `PROVINCE_SCOPE` | List of province codes to smoke-test on; set to `None` for the full 50-province run |
| `RUN_LABEL` | Suffix appended to output filenames, to avoid overwriting prior batch outputs |
| `DATETIME_RANGE` | Landsat search window |
| `MIN_PIXEL_COUNT` | Minimum valid pixels per parcel/date required to keep that observation |

---

## 3. `phase4_pilot.py` — ML classification (RF / GBDT / Logistic Regression)

**What it does:** loads the merged national parcel-level feature table, builds
two feature sets (**ET-only**: 12 zonal-stat columns; **ET+crop**: ET-only plus
one-hot encoded declared crop code), tunes RandomForest via `RandomizedSearchCV`
(or reuses a fixed param set — see below), evaluates all three models on a
held-out split with a probability-threshold sweep, and runs stratified 5-fold
CV when triggered.

**Entry point:** directly runnable — `python phase4_pilot.py`. Has a proper
`if __name__ == "__main__":` block calling `main()`.

**Config to check/set before running** (top of file):
| Variable | Purpose |
|---|---|
| `RF_TUNED_PARAMS` | Set to a param dict to skip tuning and reuse it; set to `None` to re-tune from scratch |
| `FEATURE_SETS_TO_RUN` | Subset of `["ET-only", "ET+crop"]` to actually process this run |
| `FORCE_CV` | `True` = always run 5-fold CV per model, regardless of holdout overfitting gap |
| `MODEL_OUTPUT_DIR` | Where fitted models + train/test predictions/probabilities get saved via `joblib.dump`, one file per model per feature set (holdout only, not per CV fold) |

**Outputs:** printed metrics (accuracy, F1, balanced accuracy, AUC, confusion
matrix, threshold sweep) to console/log, plus persisted `.joblib` artifacts per
model in `MODEL_OUTPUT_DIR` for later reuse (e.g. regenerating ROC curves
without refitting).

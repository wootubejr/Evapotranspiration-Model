import os
import geopandas as gpd
import time
from et_model_v3 import run_et_for_aoi, raster_export

GRID_PATH = "C:\\VSCODE_Research\\ET_Model\\California_Farm.gpkg"
GRID_DF = gpd.read_file(GRID_PATH)
n = len(GRID_DF)

OUTPUT_DIR = "C:\\VSCODE_Research\\ET_Model\\et_raster"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATETIME_RANGE = "2022-07-01/2022-07-31"
WIND_SOURCE = "era5"   # or "fixed" — flip this and rerun for the ablation comparison
ERA5_CACHE_DIR = os.path.join(OUTPUT_DIR, "era5_cache")


def main():
    print(f"Processing {n} total features")
    start_time = time.time()
    for i in range(n):
        try:
            aoi_i = GRID_DF.iloc[[i]]
            out, meta = run_et_for_aoi(aoi_i, DATETIME_RANGE, wind_source=WIND_SOURCE, era5_cache_dir=ERA5_CACHE_DIR,)
            out_file = os.path.join(OUTPUT_DIR, f"ET_tile_{i}_{WIND_SOURCE}.tif")
            out_raster = raster_export(out, meta["epsg"], out_file)
            print(f"[{i+1}/{n}] Saved {out_file} (Uref={meta['Uref']:.3f} m/s, source={meta['wind_source']})")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The code executed in {elapsed_time:.4f} seconds.")
        except Exception as e:
            import traceback, sys
            print(f"Failed for tile {i}: {e}")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
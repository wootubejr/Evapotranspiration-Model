import os
import geopandas as gpd
import time
from ET_MODEL import run_et_for_aoi, raster_export

GRID_PATH = "/Volumes/PortableSSD/ET_MODEL/GRID_DATA/ALL_Continent_Grid.gpkg"
GRID_DF = gpd.read_file(GRID_PATH)
n = len(GRID_DF)

OUTPUT_DIR = "/Volumes/PortableSSD/ET_MODEL/Output/ET_tiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATETIME_RANGE = "2022-07-01/2022-07-31"
def main():
    print(f"Processing {n} total features")
    start_time = time.time()
    for i in range(n):
        try:
            aoi_i = GRID_DF.iloc[[i]]
            out, meta = run_et_for_aoi(aoi_i, DATETIME_RANGE)
            out_file = os.path.join(OUTPUT_DIR, f"ET_tile_{i}.tif")
            out_raster = raster_export(out, meta["epsg"], out_file)
            print(f"[{i+1}/{n}] Saved {out_file}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The code executed in {elapsed_time:.4f} seconds.")
        except Exception as e:
            import traceback, sys
            print(f"Failed for tile {i}: {e}")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
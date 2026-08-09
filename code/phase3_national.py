"""
Phase 3, national scale: all 50 Spanish provinces, loaded from 3 gpkg files
(alava_test.gpkg + sigpac_nacional.gpkg + sigpac_nacional_part2.gpkg), tiled with a
50km multi-zone-aware grid (build_national_grid_tiles), parcels assigned to tiles via
a vectorized spatial join (assign_parcels_to_tiles) rather than the per-tile loop the
Alava pilot used -- that loop is O(n_tiles x n_parcels) and doesn't scale to national
parcel counts. Everything downstream (date discovery, SEBAL+ERA5 run per tile/date,
zonal stats, MIN_PIXEL_COUNT filtering, date-trajectory aggregation, parc_sistexp
label) is unchanged from phase3_pilot.py -- only loading/tiling/assignment changed.

PROVINCE_SCOPE lets you smoke-test on a handful of provinces before committing to the
full 50 -- set to None for the real national run.

"""
import os
import time
import gc
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd

from et_model_v3 import (
    run_et_for_aoi,
    build_national_grid_tiles,
    assign_parcels_to_tiles,
    discover_landsat_dates,
    zonal_stats_for_parcels,
)

GPKG_FILES = [
    r"C:\VSCODE_Research\ET_Model\data\alava_test.gpkg",
    r"C:\VSCODE_Research\ET_Model\data\sigpac_nacional.gpkg",
    r"C:\VSCODE_Research\ET_Model\data\sigpac_nacional_part2.gpkg",
]
OUTPUT_DIR = r"C:\VSCODE_Research\ET_Model\phase3_national"

RUN_LABEL = "batch_4"  # e.g. "" for the original run
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROVINCE_SCOPE = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

CELL_SIZE_M = 50000
DATETIME_RANGE = "2025-07-01/2025-07-31"
WIND_SOURCE = "era5"
ERA5_CACHE_DIR = os.path.join(OUTPUT_DIR, "era5_cache")
MIN_PIXEL_COUNT = 3  # see phase3_pilot.py for the real-data justification of this threshold

KEY = ["provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto"]
ET_VARS = ("ET_inst", "ET24h", "LE", "EF", "NDVI", "NDWI")


def load_national_parcels():
    # When PROVINCE_SCOPE is set, push the filter down into the read itself via
    # GDAL's `where` clause to avoid reading all 50 provinces into memory first.
    where_clause = None
    if PROVINCE_SCOPE is not None:
        where_clause = f"provincia IN ({','.join(str(p) for p in PROVINCE_SCOPE)})"
        print(f"PROVINCE_SCOPE={PROVINCE_SCOPE}: pushing filter to disk read (where: {where_clause})")

    all_merged = []
    for path in GPKG_FILES:
        rec = gpd.read_file(path, layer="recintos", where=where_clause)
        cd = gpd.read_file(path, layer="cultivo_declarado", where=where_clause)
        if len(rec) == 0:
            print(f"  {path}: 0 recintos in scope, skipping")
            continue
        cd_small = cd[KEY + ["parc_producto", "parc_sistexp"]]
        merged = rec.merge(cd_small, on=KEY, how="inner")
        print(f"  {path}: {len(rec)} recintos, {len(merged)} with a declared crop")
        all_merged.append(merged)

    if not all_merged:
        raise ValueError(f"PROVINCE_SCOPE={PROVINCE_SCOPE} matched 0 parcels across all 3 files -- check the codes.")

    parcels = pd.concat(all_merged, ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=all_merged[0].crs)

    # Repair invalid geometries (self-intersections, "bowties", degenerate rings) before
    # ANY geometric operation touches them.
    n_invalid = int((~parcels.geometry.is_valid).sum())
    if n_invalid:
        print(f"  Found {n_invalid} invalid geometries (self-intersections/topology "
              f"errors) -- repairing with make_valid() before proceeding.")
        parcels["geometry"] = parcels.geometry.make_valid()

        # extremely rare, but make_valid() can in principle produce a GeometryCollection
        # (polygon + stray line/point fragments) rather than a clean Polygon/MultiPolygon
        n_collections = int((parcels.geometry.geom_type == "GeometryCollection").sum())
        if n_collections:
            print(f"  WARNING: {n_collections} geometries became GeometryCollection "
                  f"(not clean Polygon/MultiPolygon) after repair -- these may cause "
                  f"errors in zonal_stats_for_parcels later. Flag for manual inspection.")

    n_before = len(parcels)

    n_null_label = parcels["parc_sistexp"].isna().sum()
    parcels = parcels[parcels["parc_sistexp"].notna()].copy()
    print(f"Dropped {n_null_label}/{n_before} parcels with no declared parc_sistexp -- "
          f"kept {len(parcels)}.")

    parcels["irrigated_label"] = (parcels["parc_sistexp"] == "R").astype(int)
    n_before_dedup = len(parcels)
    dup_mask = parcels.duplicated(subset=KEY, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        dup_keys = parcels.loc[dup_mask, KEY].drop_duplicates()
        print(f"  WARNING: {n_dup_rows} rows share a duplicate KEY across {len(dup_keys)} "
              f"distinct parcels.")

        # If parc_sistexp disagrees across a duplicate-key group, the duplication can't
        # be purely a benign multi-part-geometry split on the rec side.
        label_conflicts = parcels.loc[dup_mask].groupby(KEY)["parc_sistexp"].nunique()
        conflicting_keys = label_conflicts[label_conflicts > 1].index
        n_label_conflicts = len(conflicting_keys)

        if n_label_conflicts:
            conflict_mask = parcels.set_index(KEY).index.isin(conflicting_keys)
            n_conflict_rows = int(conflict_mask.sum())
            print(f"  EXCLUDING {n_label_conflicts} parcels ({n_conflict_rows} rows) with "
                  f"DISAGREEING parc_sistexp across duplicate declarations -- label is "
                  f"ambiguous for these, so they're dropped rather than guessed. Worth "
                  f"spot-checking a few of these specific keys in QGIS out of curiosity, "
                  f"but not worth including in training.")
            clean_dup_mask = dup_mask & ~conflict_mask
            parcels = parcels[~conflict_mask].copy()
            dup_mask = clean_dup_mask[~conflict_mask]
        else:
            print(f"  Good news: parc_sistexp agrees across all duplicate rows per parcel "
                  f"-- consistent with a benign geometry-only split (dissolve is safe here).")

        if dup_mask.any():
            unique_part = parcels.loc[~dup_mask]
            dup_part = parcels.loc[dup_mask]
            dissolved = dup_part.dissolve(by=KEY, aggfunc="first").reset_index()
            parcels = pd.concat([unique_part, dissolved], ignore_index=True)
            parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=unique_part.crs)
        print(f"  After conflict exclusion + dissolve: {n_before_dedup} -> {len(parcels)} unique parcels")

    parcels["parcel_uid"] = parcels[KEY].astype(str).agg("_".join, axis=1)

    print(f"National parcel set: {len(parcels)} total "
          f"({parcels['irrigated_label'].sum()} irrigated / {(parcels['irrigated_label']==0).sum()} rainfed)")
    return parcels


def main():
    print("Loading national parcels from 3 gpkg files...")
    parcels = load_national_parcels()

    print(f"\nBuilding {CELL_SIZE_M/1000:.0f}km multi-zone national grid...")
    grid = build_national_grid_tiles(parcels, cell_size_m=CELL_SIZE_M)
    print(f"{len(grid)} tiles across all UTM zones present in scope")

    print("\nAssigning parcels to tiles via spatial join...")
    mapping = assign_parcels_to_tiles(parcels, grid, key_cols=["parcel_uid"])
    parcels_with_tile = parcels.merge(mapping, on="parcel_uid", how="inner")

    n_before_tile_dedup = len(parcels_with_tile)
    parcels_with_tile = parcels_with_tile.drop_duplicates(subset="parcel_uid", keep="first")
    n_boundary_dupes = n_before_tile_dedup - len(parcels_with_tile)
    if n_boundary_dupes:
        print(f"  WARNING: {n_boundary_dupes} parcels matched more than one tile "
              f"(likely exact tile-boundary edge cases) -- kept first match only.")

    n_unassigned = len(parcels) - len(parcels_with_tile)
    if n_unassigned:
        print(f"  WARNING: {n_unassigned} parcels did not fall inside any tile "
              f"(check grid coverage / CRS alignment)")
    print(f"{parcels_with_tile['parcel_uid'].nunique()} parcels assigned across "
          f"{parcels_with_tile['tile_id'].nunique()} occupied tiles")

    all_stats = []
    total_dropped_pixcount = 0
    start_time = time.time()

    occupied_tile_ids = parcels_with_tile["tile_id"].unique()
    grid_by_id = grid.set_index("tile_id")

    n_tiles_total = len(occupied_tile_ids)
    for i, tile_id in enumerate(occupied_tile_ids):
        elapsed_min = (time.time() - start_time) / 60
        print(f"\n{'='*60}\nTile {i+1}/{n_tiles_total}  (tile_id={tile_id})  [{elapsed_min:.1f} min elapsed]\n{'='*60}")

        tile_geom = grid_by_id.loc[tile_id, "geometry"]
        tile_gdf = gpd.GeoDataFrame([{"tile_id": tile_id, "geometry": tile_geom}], geometry="geometry", crs=grid.crs)
        tile_parcels = parcels_with_tile[parcels_with_tile["tile_id"] == tile_id]

        bbox_wgs84 = tile_gdf.to_crs(4326).total_bounds.tolist()
        try:
            dates = discover_landsat_dates(bbox_wgs84, DATETIME_RANGE)
        except RuntimeError:
            print(f"[tile {tile_id}] ({i+1}/{n_tiles_total}) no cloud-free scenes, skipping")
            continue
        print(f"[tile {tile_id}] ({i+1}/{n_tiles_total}) {len(tile_parcels)} parcels, "
              f"{len(dates)} candidate dates: {dates}")

        for date in dates:
            try:
                out, meta = run_et_for_aoi(
                    tile_gdf, f"{date}/{date}",
                    wind_source=WIND_SOURCE, era5_cache_dir=ERA5_CACHE_DIR,
                )
                stats = zonal_stats_for_parcels(out, meta["epsg"], tile_parcels, id_cols=KEY, vars=ET_VARS)

                pixcount_cols = [f"{v}_pixcount" for v in ET_VARS]
                min_pixcount = stats[pixcount_cols].min(axis=1)
                n_before = len(stats)
                stats = stats[min_pixcount > MIN_PIXEL_COUNT].copy()
                n_dropped = n_before - len(stats)
                total_dropped_pixcount += n_dropped

                stats["date"] = date
                stats["tile_id"] = tile_id
                stats["Uref"] = meta["Uref"]
                all_stats.append(stats)
                print(f"  [{date}] ok, {n_dropped}/{n_before} parcels dropped (<={MIN_PIXEL_COUNT} px), "
                      f"Uref={meta['Uref']:.2f} m/s ({time.time()-start_time:.0f}s elapsed)")

                # Explicit cleanup: `out` holds the full-resolution 6-variable dataset
                # (materialized in-memory by zonal_stats_for_parcels for speed -- see
                # methodology_notes.md). Don't rely on implicit garbage collection
                # timing across a long unattended multi-hour run; free it now. (`stats`
                # is NOT deleted here.
                del out
                gc.collect()
            except Exception as e:
                print(f"  [{date}] FAILED: {e}")
                traceback.print_exc()
                continue

    if not all_stats:
        print("No successful tile/date runs -- nothing to aggregate.")
        return

    date_level = pd.concat(all_stats, ignore_index=True)
    print(f"\nTotal parcel/date observations dropped for <={MIN_PIXEL_COUNT} valid pixels: {total_dropped_pixcount}")
    date_level_path = os.path.join(OUTPUT_DIR, f"phase3_national_date_level{'_' + RUN_LABEL if RUN_LABEL else ''}.csv")
    date_level.to_csv(date_level_path, index=False)
    print(f"Saved date-level table ({len(date_level)} rows) -> {date_level_path}")

    mean_cols = [c for c in date_level.columns if c.endswith("_mean")]
    parcel_level = date_level.groupby(KEY)[mean_cols].agg(["mean", "std"])
    parcel_level.columns = ["_".join(c) for c in parcel_level.columns]
    parcel_level = parcel_level.reset_index()

    label_cols = KEY + ["parc_sistexp", "irrigated_label", "coef_regadio", "parc_producto"]
    parcel_level = parcel_level.merge(parcels_with_tile[label_cols].drop_duplicates(subset=KEY), on=KEY, how="left")

    out_path = os.path.join(OUTPUT_DIR, f"phase3_national_parcel_level{'_' + RUN_LABEL if RUN_LABEL else ''}.csv")
    parcel_level.to_csv(out_path, index=False)
    print(f"Saved parcel-level feature table ({len(parcel_level)} parcels) -> {out_path}")

    n_irrigated = int(parcel_level["irrigated_label"].sum())
    n_rainfed = len(parcel_level) - n_irrigated
    print(f"\nFinal class balance: {n_irrigated} irrigated / {n_rainfed} rainfed "
          f"({parcel_level['irrigated_label'].mean():.1%} irrigated, {len(parcel_level)} total)")


if __name__ == "__main__":
    main()

#package import
import math, warnings, os, time

# Bound GDAL's own internal caching BEFORE rasterio/stackstac (both GDAL-backed) are
# imported below -- these env vars only take effect if set before GDAL initializes.
# This addresses a known, documented failure mode for long-running loops that open
# many DIFFERENT remote Cloud-Optimized GeoTIFFs (one new tile/date = new STAC items
# = new remote files every iteration): GDAL's VSICURL remote-read cache is a C-level
# cache, invisible to and uncontrollable by Python's gc.collect() -- it can grow
# unboundedly across hundreds of distinct file opens in a single long-lived process.
# Confirmed real for this pipeline: Windows Event Viewer logged python.exe consuming
# 112GB virtual memory before a crash, in a run that processed hundreds of tiles each
# pulling fresh remote Landsat/DEM assets.
os.environ.setdefault("GDAL_CACHEMAX", "512")               # cap GDAL's raster block cache at 512MB
os.environ.setdefault("VSI_CACHE", "FALSE")                 # disable per-file VSI caching entirely -- the main suspect for unbounded growth across many distinct remote files
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "16000000")  # bound the CURL-level remote-read cache (~16MB) as a second layer, in case VSI_CACHE=FALSE alone isn't enough
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")  # standard cloud-COG best practice, avoids unnecessary remote directory-listing overhead/caching
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
import stackstac, pystac_client, planetary_computer as pc
from shapely.geometry import box
from pyproj import CRS
from pyproj import Transformer
from timezonefinder import TimezoneFinder
import pvlib
import dask
import dask.array as dka
#from concurrent.futures import ThreadPoolExecutor, as_completed
#from rasterio.windows import Window
warnings.filterwarnings("ignore", category=RuntimeWarning)

#tools
dask.config.set({
    "scheduler": "threads",
    "num_workers": os.cpu_count(),
    "array.slicing.split_large_chunks": True,   
    "optimization.fuse.active": True,         
})

Solar_Const = 1361.0
boltzmann = 5.670374419e-8
LAPSE = 0.0065 

def safe_div(num, den):
    return num/ xr.where(den != 0, den, np.nan)

def safe_pow(x, p):
    base = xr.where(np.isfinite(x), x, np.nan)
    if isinstance(p, (float, np.floating)) and (abs(p - int(p)) > 1e-12):
        base = xr.where(base >= 0, base, np.nan)
    return xr.apply_ufunc(np.power, base, p, dask="parallelized",
                          output_dtypes=[np.float32])

def safe_log(x):
    arg = xr.where(np.isfinite(x) & (x > 0), x, np.nan)
    return xr.apply_ufunc(np.log, arg, dask="parallelized",
                          output_dtypes=[np.float32])

def q_scalar(da, q):
    return (
        da.quantile(q, dim=("y", "x"), skipna=True, method="nearest")
        .squeeze(drop=True)
        .drop_vars("quantile", errors="ignore")
    )

def utm_epsg_from_lonlat(lon: float, lat: float):
    
    lon_norm = ((lon + 180) % 360) - 180
    zone = int(np.floor((lon_norm + 180) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    if lat < 0:
        raise ValueError(f"lat={lat} is in the southern hemisphere -- this pipeline is Spain-only (ETRS89/UTM has no south zones).")
    if 28 <= zone <= 38:
        epsg_num = 25800 + zone
    else:
        epsg_num = 32600 + zone  # WGS84 fallback -- ETRS89 has no code for this zone (e.g. Canary Islands' western extent)
    return f"EPSG:{epsg_num}", zone, lon_norm

def find_timezone(lon: float, lat:float) -> str: 
    tz = TimezoneFinder().timezone_at(lng=float(lon), lat=float(lat))
    return tz or "UTC"

#ERA5 wind 
ERA5_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_GRID_DEG = 0.25 

_era5_ds_singleton = {"ds": None}

def _open_era5_dataset():
    """Opens (once per process) and caches the lazy ERA5 zarr handle. Opening is cheap
    (reads only metadata); the expensive part is the subsequent .sel() data pull."""
    if _era5_ds_singleton["ds"] is None:
        print("[ERA5] opening zarr store (metadata only)...")
        _era5_ds_singleton["ds"] = xr.open_zarr(
            ERA5_ZARR_PATH,
            chunks=None,
            storage_options=dict(token="anon"),
        )
    return _era5_ds_singleton["ds"]

def _era5_cache_path(cache_dir, lon_key, lat_key, ts_hour):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"era5_wind_{lon_key:.2f}_{lat_key:.2f}_{ts_hour.strftime('%Y%m%dT%H')}.json"
    return os.path.join(cache_dir, fname)

def get_era5_uref(lon: float, lat: float, timestamp, cache_dir: str = "era5_cache") -> float:
    """Uref = sqrt(u10**2 + v10**2) at the AOI centroid, nearest ERA5 hourly step to
    `timestamp` (Landsat overpass time). Caches to a small local JSON file keyed on the
    ERA5 grid cell + hour so repeated main.py runs over the same AOI/date don't re-hit
    the bucket. zref is left at 10.0 by the caller since these are the 10m wind fields."""
    import json

    ts = pd.Timestamp(timestamp)
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    ts_hour = ts.round("h").tz_localize(None)  

    lon_360 = float(lon) % 360.0
    lon_key = round(lon_360 / ERA5_GRID_DEG) * ERA5_GRID_DEG
    lat_key = round(float(lat) / ERA5_GRID_DEG) * ERA5_GRID_DEG

    cache_file = _era5_cache_path(cache_dir, lon_key, lat_key, ts_hour)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        return float(cached["Uref"])

    ds = _open_era5_dataset()
    point = ds[["10m_u_component_of_wind", "10m_v_component_of_wind"]].sel(
        time=ts_hour, longitude=lon_key, latitude=lat_key, method="nearest"
    )
    point = point.compute()
    u10 = float(point["10m_u_component_of_wind"].values)
    v10 = float(point["10m_v_component_of_wind"].values)
    Uref = float(np.sqrt(u10**2 + v10**2))

    with open(cache_file, "w") as f:
        json.dump(
            {
                "lon": lon, "lat": lat, "lon_360": lon_360,
                "requested_time": str(ts), "matched_hour_utc": str(ts_hour),
                "u10": u10, "v10": v10, "Uref": Uref,
            },
            f,
        )
    return Uref

def make_gdal_env():
    return rio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="YES",
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_HTTP_MAX_RETRY="8",
        GDAL_HTTP_RETRY_DELAY="2",
        GDAL_HTTP_RETRY_CODES="206,429,500,502,503,504",
        GDAL_HTTP_CONNECTTIMEOUT="10",
        GDAL_HTTP_TIMEOUT="240",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MULTIRANGE="YES",
        CPL_VSIL_CURL_CHUNK_SIZE=str(16 * 1024 * 1024),  # 16 MB
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=str(512 * 1024 * 1024),
    )

#Stack search + stack
def search_sign_items(bbox_wgs84, datetime=None, collections=None, query=None):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    print('Searching catalog...')
    kwargs = {
        "collections": collections,
        "bbox": bbox_wgs84,
    }
    if datetime is not None:
        kwargs["datetime"] = datetime
    if query is not None:
        kwargs["query"] = query

    search = catalog.search(**kwargs
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No items found for AOI/time.")
    return [pc.sign(it) for it in items]


def stack_assets(items, assets, epsg, bbox_utm, res=30, chunksize=2048):
    print("Stacking tiles")
    epsg_int = int(epsg.split(":")[1])
    stack = stackstac.stack(
        items,
        assets = assets,
        resolution=res,
        epsg=epsg_int,
        bounds=bbox_utm,
        chunksize=chunksize,
    errors_as_nodata=(Exception,)
    )
    stack.rio.write_crs(epsg, inplace=True)
    return stack

def stack_and_composite(items, required_bands, epsg, bbox_utm, **kw):
    stack = stack_assets(items, required_bands, epsg, bbox_utm, **kw)
    return stack.median("time", keep_attrs=True)

def bounds_to_wgs84(bounds_xy, epsg_from):
    xmin, ymin, xmax, ymax = bounds_xy
    t = Transformer.from_crs(epsg_from, "EPSG:4326", always_xy=True)
    minx, miny = t.transform(xmin, ymin)
    maxx, maxy = t.transform(xmax, ymax)
    return [minx, miny, maxx, maxy]

def pad_bounds(bounds_xy, pad_m):
    xmin, ymin, xmax, ymax = bounds_xy
    return [xmin - pad_m, ymin - pad_m, xmax + pad_m, ymax + pad_m]

def build_dem(dem_items, epsg: str, bbox_utm: list, match_da: xr.DataArray) -> xr.DataArray:
    if match_da.rio.crs is None or match_da.rio.transform() is None:
        raise ValueError("match_da must have CRS and transform")
    dem_stack = stack_assets(dem_items, ["data"], epsg, bbox_utm, res=60, chunksize=1024)
    nodata = dem_stack.rio.nodata
    if nodata is not None:
        dem_stack = dem_stack.where(dem_stack != nodata)
    dem_2d = (
        dem_stack.max("time", skipna=True)
                 .isel(band=0)
                 .squeeze(drop=True)
                 .drop_vars("band", errors="ignore")
                 .rio.write_crs(epsg)
                 .rename("DEM")
                 .astype("float32")
                 .chunk({"y": 1024, "x": 1024})
    )
    dem_match = (
        dem_2d.rio.reproject_match(
            match_da, 
            resampling=Resampling.nearest, 
            nodata=np.nan
            )
            .astype("float32")
            .chunk({"y": 1024, "x": 1024})
            .rename("DEM")
    )
    return dem_match

#indices function (1)
def add_spectral_indices(ds):
    R, G, B, NIR, SWIR1, SWIR2, LWIR = (
    ds["red"], ds["green"], ds["blue"],
    ds["nir08"], ds["swir16"], ds["swir22"],
    ds["lwir11"],)
    ndvi = safe_div(NIR - R, NIR + R).clip(-1, 1).rename("NDVI")
    savi = safe_div(1.5*(NIR-R),0.5+NIR+R).clip(-1,1).rename("SAVI")
    ndwi = safe_div(G-NIR, G+NIR).clip(-1,1).rename("NDWI")
    savi1 = savi < 0.689
    log_NumDem = (0.69-savi).clip(1e-6)
    log_set = np.log(log_NumDem) / 0.59
    lai = safe_div(-log_set, 0.91).clip(0,7).rename("LAI")
    emissivity = (0.95 + 0.01 * lai)
    emissivity = emissivity.clip(0.95, 0.99)
    albedo = 0.4739*B-0.4372*G+0.1652*R+0.2831*NIR+0.1072*SWIR1+0.1029*SWIR2+0.0366

    return ds.assign(
        NDVI=ndvi,
        SAVI=savi,
        NDWI = ndwi,
        LAI=lai,
        e_0 = emissivity,
        albedo = albedo
    )

#Radiation functions (2)

#LAND SURFACE TEMPERATURE WITH DEM CORRECTION AND ASPECT/SLOPE
def compute_radiation(img, sun_elev, sun_azim):
    DEM   = img["DEM"]
    albedo= img["albedo"]
    BB_emissivity = img["e_0"]
    LSTk  = img["lwir11"]
    NDVI = img["NDVI"]
    
    sun_elev_deg = float(sun_elev)
    sun_azim_deg = float(sun_azim)
    solar_zen_angle = np.deg2rad(90.0 - sun_elev_deg)
    gamma_s = np.deg2rad(sun_azim_deg)

    #elv_ref = float(DEM.mean().compute()) return to this is this sparse sample doesn't work well enough
    elv_ref = float(DEM.isel(y=slice(None, None, 64), x=slice(None, None, 64)).mean().compute())
    LSTk_elv_corr = (LSTk + LAPSE*(elv_ref - DEM)).rename("LSTK_demcorr")

    DEM = DEM.chunk({"y": 2048, "x": 2048}).astype("float32")
    xres = float(abs(DEM["x"][1] - DEM["x"][0]))
    yres = float(abs(DEM["y"][1] - DEM["y"][0]))
    gdy, gdx = dka.gradient(DEM.data, yres, xres)
    dz_dy, dz_dx = xr.DataArray(gdy, coords = DEM.coords, dims=DEM.dims), xr.DataArray(gdx, coords = DEM.coords, dims=DEM.dims)
    slope = xr.apply_ufunc(np.arctan, dka.hypot(dz_dx.data, dz_dy.data), dask="allowed")         
    aspect = xr.apply_ufunc(np.arctan2, -dz_dy.data, -dz_dx.data, dask="allowed")       
    aspect = (aspect + 2*np.pi) % (2*np.pi)

    cos_theta_z = np.cos(solar_zen_angle); sin_theta_z = np.sin(solar_zen_angle)
    cos_beta = np.cos(slope);   sin_beta    = np.sin(slope)
    cos_dgamma = np.cos(gamma_s - aspect)
    cos_i = (cos_theta_z*cos_beta) + (sin_theta_z*sin_beta*cos_dgamma)

    TOA_irridance_SolarNoon = Solar_Const * max(np.cos(solar_zen_angle), 0.0)
    tau_clear = (0.75 + 2.0e-5 * DEM).clip(0.55, 0.90)

    SW_cs = tau_clear * (TOA_irridance_SolarNoon * (cos_i/ max(np.cos(solar_zen_angle), 1e-6)))

    NET_SW = ((1.0 - albedo) * SW_cs)

    Ta = LSTk_elv_corr

    emiss_clear_sky = (0.72 + 1.6e-3*(Ta-273.15)).clip(0.72, 0.95)
    L_up = (BB_emissivity * boltzmann * safe_pow(LSTk_elv_corr,4)).rename("L_up")
    L_down = (emiss_clear_sky * boltzmann * safe_pow(Ta,4)).rename("L_down")
    NET_LW = (L_down - L_up)

    NET_RAD = (NET_SW +NET_LW).where(np.isfinite(NET_SW + NET_LW)).rename("Rn")

    return img.assign(
    LSTk_demcorr=LSTk_elv_corr, 
    Rs_cs= SW_cs,
    Rns=NET_SW,
    Ta = Ta,
    L_up=L_up,
    L_down=L_down,
    Rn=NET_RAD)

#Flux functions (3)

def compute_fluxes(image, Uref=3.5, zref=10.0):

    NDVI, LAI, ALBEDO = image["NDVI"], image["LAI"], image["albedo"]
    NIR, SWIR1        = image["nir08"], image["swir16"]
    Rn, TsK, DEM      = image["Rn"], image["LSTk_demcorr"], image["DEM"]
    TsC = TsK - 273.15
    NDVI = NDVI.astype("float32"); Rn = Rn.astype("float32"); TsK = TsK.astype("float32")

    G_sebs = Rn * (TsC.clip(0,30) * (0.0038 + 0.006 * ALBEDO) * (1 - 0.995 * (NDVI ** 4)))
    G_lai = Rn * (0.05 + 0.18 * np.exp(-0.521 * LAI)).clip(0.02, 0.35)

    NDVI_clean = NDVI.astype("float32").where(np.isfinite(NDVI))
    NDVI_quant = NDVI_clean.quantile([0.05, 0.95], dim=("y","x"), skipna=True, method="nearest")
    ndvi_min = NDVI_quant.sel(quantile=0.05)
    ndvi_max = NDVI_quant.sel(quantile=0.95)
    denom = (ndvi_max - ndvi_min).clip(min=1e-6)
    fc = ((NDVI_clean - ndvi_min) / denom).clip(0, 1) **2 #squared to emphasise dense canopy
    G_bare = (0.24+0.18*ALBEDO).clip(0.20,0.40)*Rn
    G_veg  = (0.05 + 0.02 *np.exp(-0.7*LAI)).clip(0.03,0.12) * Rn  
    G_mix  = G_bare * (1 - fc) + G_veg * fc
    G0 = xr.where(fc >= 0.15, 0.5 * (G_sebs + G_lai), G_mix)
    G_inst = G0.clip(0, Rn * 0.6).rename("G_inst")

    NDMI = safe_div((NIR-SWIR1), (NIR+SWIR1)).astype("float32")
    NDMI_clean = NDMI.astype("float32").where(np.isfinite(NDMI))
    NDMI_quant = NDMI_clean.quantile([0.05, 0.95], dim=("y","x"), skipna=True, method="nearest")
    ndmi_min = NDMI_quant.sel(quantile= 0.05)
    ndmi_max = NDMI_quant.sel(quantile= 0.95)
    denom_ndmi = (ndmi_max - ndmi_min).clip(min=1e-6)
    m = ((NDMI_clean - ndmi_min) / denom_ndmi).clip(0, 1)
    k = 0.35
    G = G_inst *(1-k*m)
    G_inst = G.clip(0, Rn*0.6).rename("G_inst")

    air_heat = 1004

    Karman = 0.41

    canopy_height = (0.1 + 2.0 * fc).clip(0.05, 3.0)

    canopy_displace = (0.67 * canopy_height).clip(0.0)                   

    SAVI = image["SAVI"]
    z0m = xr.apply_ufunc(np.exp, 5.62*SAVI - 5.809,
                     dask="parallelized", output_dtypes=[np.float64]).clip(0.005, 0.5)

    zT = xr.zeros_like(canopy_height) + 2.0
    zH = xr.zeros_like(canopy_height) + 0.1

    psiU   = safe_log((zref - canopy_displace).clip(min=0.05) / z0m.clip(min=1e-6))
    ustar  = (Karman * Uref) / psiU
    ustar  = ustar.clip(0.05, 2.0)           

    psiH = safe_log((zT - canopy_displace).clip(min=0.05)  / (zH - canopy_displace).clip(min=0.02))
    rah0 = psiH / (Karman * ustar)
    rah0 = rah0.clip(20.0, 200.0).rename("rah0_neutral")

    rah_start = rah0.copy()             
    ufric = ustar.copy()      
    
    #========ITERATIVE PROCESS=========#

    n_dif= 1
    list_dif = []
    list_dT_hot = []
    list_rah_hot = []
    list_coef_a = []
    list_coef_b = []

    Ts = TsK
    NDWI = image["NDWI"]
    DEM = image['DEM']
    dry_air_gas  = 287.0                  
    Ta = image['Ta']
    p = 101325.0 * ((1.0 - 2.25577e-5 * DEM).clip(min=1e-6))**5.2559 
    rho = p / (dry_air_gas * Ta)  
    
    veg_thresh = 0.7 
    bare_thresh = 0.2

    cand_cold = Ts.notnull() & (NDVI >= veg_thresh) & (NDWI < 0.1) & (ALBEDO >= 0.08) & (ALBEDO <= 0.40)
    cand_hot = Ts.notnull() & (NDVI <= bare_thresh) & (NDWI < 0.0)
  
    Q = xr.Dataset({
        "Ts_hot":  q_scalar(Ts.where(cand_hot),   0.98),
        "Ts_cold": q_scalar(Ts.where(cand_cold),  0.05),
        "Rn_hot":  q_scalar(Rn.where(cand_hot),   0.90),
        "G_hot":   q_scalar(G_inst.where(cand_hot), 0.50),
        "rho_hot": q_scalar(rho.where(cand_hot),  0.50),
    })

    vals = Q.compute()
    Ts_hot  = float(vals["Ts_hot"].item())
    Ts_cold = float(vals["Ts_cold"].item())
    Rn_hot  = float(vals["Rn_hot"].item())
    G_hot   = float(vals["G_hot"].item())
    rho_hot = float(vals["rho_hot"].item())

    dT_int = xr.full_like(Ts, np.nan)

    rho = rho
    z0m = z0m
    G_inst = G_inst

    rah = rah_start
    rah_hot = np.nan
    recompute_iters = {1, 3, 5, 7}
    #========INIT ITERATION========#
    for it in range(1, 8):
        if it in recompute_iters:
            rah_hot = float(
                rah.where(cand_hot)
                    .quantile(0.80, dim=("y","x"), skipna=True, method="nearest")
                    .compute()
                    .item()
            )

        dT_hot   = np.nan
        denom_ts = np.nan
        a = np.nan
        b = np.nan

        H_hot = max(Rn_hot - G_hot, 0.0)
        
        if np.isfinite(H_hot) and (H_hot > 10) and np.isfinite(rah_hot) and np.isfinite(rho_hot):
            dT_hot = (H_hot * rah_hot) / (rho_hot * air_heat)
            dT_hot = float(np.clip(dT_hot, 5.0, 25.0))
            denom_ts = Ts_hot - Ts_cold
        if np.isfinite(denom_ts) and (abs(denom_ts) >= 0.2):
            a = (dT_hot - 0.5) / denom_ts
            b = 0.5 - a * Ts_cold
        
        dT_int = xr.where(np.isfinite(a) & np.isfinite(b) & np.isfinite(Ts), a * Ts + b, dT_int)
        H_int  = xr.where(np.isfinite(dT_int), (rho * air_heat * dT_int) / xr.where(rah > 1e-3, rah, np.nan), np.nan)

        MO_L_int = -1 * safe_div(rho*air_heat*(ufric**3)*Ts, (Karman*9.81*H_int))
        MO_L_int = xr.where(np.abs(MO_L_int) < 1.0, np.sign(MO_L_int)*1.0, MO_L_int)

        expr200 = 1-16*(200/MO_L_int)
        x200 = safe_pow(expr200, 0.25)
        expr2 = 1-16*(2/MO_L_int)
        x2 = safe_pow(expr2, 0.25)
        expr01 = 1-16*(0.01/MO_L_int)
        x01 = safe_pow(expr01, 0.25)

        psimu_200 = (2*safe_log((1+x200)/2)+safe_log((1+safe_pow(x200,2))/2)-2*np.atan(x200)+0.5*3.14159265)
        psihu_2 = 2*safe_log((1+safe_pow(x2,2))/2)
        psihu_01 = 2*safe_log((1+safe_pow(x01,2))/2)

        psi_m_stab_200 = -5*(200/MO_L_int)
        psi_m_stab_2 = -5*(2/MO_L_int)
        psi_m_stab_01 = -5*(0.01/MO_L_int)

        psim_200 = xr.where(MO_L_int < 0, psimu_200, xr.where(MO_L_int > 0, psi_m_stab_200, 0))
        psih_2 = xr.where(MO_L_int < 0, psihu_2, xr.where(MO_L_int > 0, psi_m_stab_2, 0))
        psih_01 = xr.where(MO_L_int < 0, psihu_01, xr.where(MO_L_int > 0, psi_m_stab_01, 0))

        den_fric = (np.log(zref/z0m) - psim_200)
        ufric = (Uref*Karman) / xr.where(np.abs(den_fric) < 1e-3,
                                 xr.where(den_fric>=0, 1e-3, -1e-3),
                                 den_fric)

        rah = (np.log(zT/zH)-psih_2+psih_01)/(ufric*Karman)
        rah = rah.clip(10.0, 500.0)

        if it==1:
            n_dT_hot_old  = dT_hot if np.isfinite(dT_hot) else 0.0
            n_rah_hot_old = rah_hot if np.isfinite(rah_hot) else 0.0
            n_dif = 1.0
        else:
            dif_dT  = (dT_hot  - n_dT_hot_old)  if (np.isfinite(dT_hot)  and np.isfinite(n_dT_hot_old))  else 0.0
            dif_rah = (abs(rah_hot) - abs(n_rah_hot_old)) if (np.isfinite(rah_hot) and np.isfinite(n_rah_hot_old)) else 0.0
            n_dif   = abs(dif_dT) + abs(dif_rah)
            n_dT_hot_old  = dT_hot
            n_rah_hot_old = rah_hot

        list_dif.append(float(n_dif))
        list_coef_a.append(float(a) if np.isfinite(a) else np.nan)
        list_coef_b.append(float(b) if np.isfinite(b) else np.nan)
        list_dT_hot.append(float(dT_hot) if 'dT_hot' in locals() and np.isfinite(dT_hot) else np.nan)
        list_rah_hot.append(float(rah_hot) if np.isfinite(rah_hot) else np.nan)

    #=========END ITERATION =========#

    Available_energy = (Rn - G_inst).astype("float32")
    rah_final = xr.where(np.isfinite(rah), rah, np.nan).astype("float32").rename("rah")
    dT_final  = xr.where(np.isfinite(dT_int), dT_int, np.nan).astype("float32").rename("dT")
    rah_safe = xr.where(rah_final > 1e-3, rah_final, np.nan)
    H_raw = (rho * air_heat * dT_final) / rah_safe
    Ra_pos = xr.where(Available_energy > 0, Available_energy, 0)
    H_cap  = Ra_pos * 0.98
    H_final = xr.where(Ra_pos > 0, H_raw.clip(0, H_cap), 0).rename("H").astype("float32")
    
    return image.assign(
            rah_final = rah_final,
            dT = dT_final,
            H = H_final,
            zom= z0m,
            i_ufric = ufric,
            G = G_inst)

#ET Functions (4)

def compute_ET(image, lat, lon, tz, datetime_range):
    Rn = image["Rn"] # daily net radiation Wm^-2
    G = image["G"] #soil/ground heat flux Wm^-2
    LST = image["lwir11"] #[K]
    H = image["H"] #sensible heat flux (Wm^-2)
    H=xr.where(H< 0, 0, H)

    #LATENT HEAT FLUX (LE) [W M-2]
    LE = (Rn-G-H)
    LE = xr.where(LE < 0, 0, LE)

    lambda_MJkg = (2.501-0.002361*(LST-273.15)) 
    lambda_Jkg = (2.501e6 - 2361*(LST - 273.15)) 

    #INSTANTANEOUS ET [MM H-1]
    ET_inst = 0.0036*(LE/lambda_MJkg)
  
    A_inst = (Rn - G).clip(min=0)
    EF = safe_div(LE, xr.where(A_inst>0, A_inst, np.nan)).clip(0,1)
   
    #DAILY EVAPOTRANSPIRATION (ET_24h) [MM DAY-1]
    start_s, end_s = datetime_range.split("/")
    start = pd.Timestamp(start_s, tz=tz)
    end   = pd.Timestamp(end_s, tz=tz)
    mid   = start + (end - start) / 2
    times = pd.DatetimeIndex([mid])
    sr_ss = pvlib.solarposition.sun_rise_set_transit_spa(times, lat, lon).iloc[0]
    sunrise, sunset = sr_ss["sunrise"], sr_ss["sunset"]
    daylength_seconds = (sunset - sunrise).total_seconds()

    dirunal = 0.75 
    LE_day_Jm2 = LE * daylength_seconds * dirunal #J m-2 day-1
    ET24h = (LE_day_Jm2 / lambda_Jkg) 

    return image.assign(ET_inst = ET_inst, ET24h = ET24h, LE = LE, EF = EF)


def raster_export(image: xr.Dataset, epsg: str, out_path: str) -> xr.DataArray:
    vars_to_write = ["LE", "EF", "ET_inst", "ET24h"]
    export_ds = image[vars_to_write].astype("float32")
    da = export_ds.to_array(dim="band")

    da = da.rio.write_crs(epsg, inplace=False)
    da = da.rio.write_nodata(np.nan, inplace=False)

    da.rio.to_raster(
        out_path,
        compress="deflate",
        tiled=True,
        blockxsize=2048, #potentiall change x and y back to 256 if data quality breaks
        blockysize=2048,
        BIGTIFF="IF_SAFER",
        dtype="float32",
    )
    return da

#Phase 3: AOI tiling + zonal stats

def build_grid_tiles(bounds_gdf: gpd.GeoDataFrame, cell_size_m: float = 10000, crs_utm: str = None) -> gpd.GeoDataFrame:
    """Regular square grid covering bounds_gdf's total_bounds, built in a projected CRS
    so cell_size_m is actually meters, then returned in bounds_gdf's original CRS. Each
    cell is one SEBAL AOI/tile -- large enough (10km default) to contain hot/cold pixel
    diversity for calibration, unlike a single parcel.

    If crs_utm is None (default), the correct ETRS89/UTM zone is auto-detected from
    bounds_gdf's centroid via utm_epsg_from_lonlat -- for Alava this resolves to
    EPSG:25830 automatically, and it'll resolve correctly without any code change if
    this is later reused for a province in zone 29N (Galicia) or 31N (Catalonia/
    Balearics). Note: a national-scale run spanning multiple UTM zones would still want
    per-region tiling rather than one flat grid in a single zone -- cell size distorts
    increasingly far from the chosen zone's central meridian. Not an issue at
    province scale (a single province doesn't span zones), and downstream SEBAL runs
    always reproject per-AOI to their own correct zone regardless, so grid distortion
    here would only ever affect tile *shape/size* consistency, not analysis correctness."""
    if crs_utm is None:
        wgs84_bounds = bounds_gdf.to_crs(4326).total_bounds
        cx, cy = (wgs84_bounds[0] + wgs84_bounds[2]) / 2, (wgs84_bounds[1] + wgs84_bounds[3]) / 2
        crs_utm, _, _ = utm_epsg_from_lonlat(cx, cy)

    bounds_utm = bounds_gdf.to_crs(crs_utm)
    minx, miny, maxx, maxy = bounds_utm.total_bounds
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    cells = [
        {"tile_id": i, "geometry": box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)}
        for i, (x0, y0) in enumerate((x, y) for x in xs for y in ys)
    ]
    grid = gpd.GeoDataFrame(cells, crs=crs_utm)
    return grid.to_crs(bounds_gdf.crs)

def _centroid_lon_for_zone_detection(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Longitude of each geometry's centroid, computed directly in geographic CRS.
    Normally this would be wrong (geopandas warns for good reason -- centroid math on
    lon/lat degrees is inaccurate), but zone assignment only needs ~3-degree precision
    (UTM zones are 6 degrees wide), so the geographic-CRS approximation is more than
    good enough here and reprojecting first would just cost time for no real benefit.
    Warning suppressed deliberately, not from an oversight."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        return gdf.to_crs(4326).geometry.centroid.x

def build_national_grid_tiles(parcels_gdf: gpd.GeoDataFrame, cell_size_m: float = 50000) -> gpd.GeoDataFrame:
    """Multi-zone-aware wrapper around build_grid_tiles for national coverage. Mainland
    Spain spans UTM zones 29N/30N/31N -- one flat grid in a single zone would distort
    cell size increasingly far from that zone's central meridian, which build_grid_tiles
    already warns about in its own docstring as a known limitation at this scale.

    Each parcel is assigned to its own correct UTM zone from its centroid longitude
    (zone boundaries run north-south, so longitude alone determines zone -- latitude is
    irrelevant here, unlike the true reprojected centroid used elsewhere). Parcels are
    grouped by zone, a grid is built separately per zone (each geometrically exact within
    its own zone), and the results are concatenated with zone-prefixed tile_ids (e.g.
    "25829_14") so IDs never collide across zones."""
    lon_wgs84 = _centroid_lon_for_zone_detection(parcels_gdf)
    zone_epsg = lon_wgs84.apply(lambda lon: utm_epsg_from_lonlat(lon, 40.0)[0])  # lat fixed: doesn't affect zone

    parcels_gdf = parcels_gdf.copy()
    parcels_gdf["_utm_zone"] = zone_epsg.values

    zone_grids = []
    for zone, zone_parcels in parcels_gdf.groupby("_utm_zone"):
        zone_grid = build_grid_tiles(zone_parcels, cell_size_m=cell_size_m, crs_utm=zone)
        zone_grid["tile_id"] = zone.replace("EPSG:", "") + "_" + zone_grid["tile_id"].astype(str)
        zone_grids.append(zone_grid)
        print(f"  zone {zone}: {len(zone_parcels)} parcels -> {len(zone_grid)} tiles")

    national_grid = pd.concat(zone_grids, ignore_index=True)
    return gpd.GeoDataFrame(national_grid, crs=zone_grids[0].crs)

def assign_parcels_to_tiles(parcels_gdf: gpd.GeoDataFrame, tiles_gdf: gpd.GeoDataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Vectorized spatial join (parcel centroid -> containing tile) via geopandas'
    spatial index -- replaces the per-tile .within() loop the Alava pilot used, which
    is O(n_tiles x n_parcels) and doesn't scale to national parcel counts. Centroid is
    computed in EPSG:3035 (equal-area, accurate anywhere in Spain) for correctness, same
    convention as run_et_for_aoi's centroid calc, then reprojected to match tiles_gdf's
    CRS for the actual join (containment testing is CRS-invariant as long as both
    geometries share one CRS at query time -- no need for per-zone precision here,
    unlike when the grid itself was BUILT)."""
    centroids_3035 = parcels_gdf.to_crs(3035).geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(parcels_gdf[key_cols].reset_index(drop=True), geometry=centroids_3035.values, crs=3035)
    centroids_gdf = centroids_gdf.to_crs(tiles_gdf.crs)

    joined = gpd.sjoin(centroids_gdf, tiles_gdf[["tile_id", "geometry"]], how="inner", predicate="within")
    return joined[key_cols + ["tile_id"]].reset_index(drop=True)

def select_tiles_for_municipios(grid: gpd.GeoDataFrame, parcels_gdf: gpd.GeoDataFrame, municipio_codes) -> gpd.GeoDataFrame:
    """Grid tiles that actually intersect the pilot municipios' parcels -- avoids
    processing tiles over the rest of Alava you haven't selected yet."""
    subset = parcels_gdf[parcels_gdf["municipio"].isin(municipio_codes)]
    union_geom = subset.geometry.union_all() if hasattr(subset.geometry, "union_all") else subset.geometry.unary_union
    mask = grid.geometry.intersects(union_geom)
    return grid[mask].reset_index(drop=True)

def discover_landsat_dates(bbox_wgs84, datetime_range: str, cloud_lt: float = 2) -> list[str]:
    """Distinct scene dates (YYYY-MM-DD) available for this AOI/range at the given
    cloud-cover ceiling. Raises RuntimeError (from search_sign_items) if none found --
    catch this per-tile so one dry tile doesn't kill the whole pilot loop."""
    with make_gdal_env():
        items = search_sign_items(
            bbox_wgs84, datetime_range, collections=["landsat-c2-l2"],
            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": cloud_lt}},
        )
    dates = sorted({pd.Timestamp(it.properties["datetime"]).strftime("%Y-%m-%d") for it in items})
    return dates

def zonal_stats_for_parcels(
    out_ds: xr.Dataset,
    epsg: str,
    parcels_gdf: gpd.GeoDataFrame,
    id_cols: list[str],
    vars=("ET_inst", "ET24h", "LE", "EF", "NDVI", "NDWI"),
) -> pd.DataFrame:
    """Per-parcel mean/std/pixel-count for each variable, computed in-memory straight
    off the SEBAL output dataset -- no raster written to disk. Requires `rasterstats`
    (pip install rasterstats --break-system-packages)."""
    from rasterstats import zonal_stats

    parcels_utm = parcels_gdf.to_crs(epsg)
    results = parcels_utm[id_cols].reset_index(drop=True).copy()
    print("  Materializing rasters (all vars, one compute() call)...")
    t0 = time.time()
    computed = out_ds[list(vars)].astype("float64").compute()
    print(f"  Materialized in {time.time()-t0:.1f}s")

    print("  Rasterizing parcel masks (once, shared across all vars)...")
    t0 = time.time()
    first_var = vars[0]
    dummy = np.ones_like(np.squeeze(computed[first_var].values))
    affine = computed[first_var].rio.transform()
    geom_result = zonal_stats(
        parcels_utm.geometry, dummy, affine=affine, stats=["count"],
        all_touched=False, nodata=-9999, raster_out=True,
    )
    print(f"  Rasterized in {time.time()-t0:.1f}s")

    for var in vars:
        arr = np.squeeze(computed[var].values)
        means, stds, counts = [], [], []
        for r in geom_result:
            mini_affine = r["mini_raster_affine"]
            pure_geo_mask = r["mini_raster_array"].mask
            h, w = pure_geo_mask.shape
            col_off = round((mini_affine.c - affine.c) / affine.a)
            row_off = round((mini_affine.f - affine.f) / affine.e)

            row_start, row_end = max(0, row_off), min(arr.shape[0], row_off + h)
            col_start, col_end = max(0, col_off), min(arr.shape[1], col_off + w)

            if row_end <= row_start or col_end <= col_start:
                means.append(np.nan); stds.append(np.nan); counts.append(0)
                continue

            window = arr[row_start:row_end, col_start:col_end]
            mask_window = pure_geo_mask[
                row_start - row_off: row_start - row_off + (row_end - row_start),
                col_start - col_off: col_start - col_off + (col_end - col_start),
            ]
            combined_mask = mask_window | np.isnan(window)
            masked = np.ma.masked_array(window, mask=combined_mask)
            cnt = int(masked.count())
            means.append(float(masked.mean()) if cnt > 0 else np.nan)
            stds.append(float(masked.std()) if cnt > 0 else np.nan)
            counts.append(cnt)
        results[f"{var}_mean"] = means
        results[f"{var}_std"] = stds
        results[f"{var}_pixcount"] = counts

    return results

#Entry point main.py
def run_et_for_aoi(
    aoi: gpd.GeoDataFrame,
    datetime_range: str,
    wind_source: str = "fixed",      # "fixed" or "era5"
    Uref_fixed: float = 3.5,
    zref: float = 10.0,
    era5_cache_dir: str = "era5_cache",
):

    aoi_proj = aoi.to_crs(epsg=3035)
    aoi_wgs84 = aoi.to_crs(epsg=4326)
    centroid_proj = aoi_proj.geometry.centroid 
    centroid_wgs84 = gpd.GeoSeries(centroid_proj, crs="EPSG:3035").to_crs("EPSG:4326")
    print("[CENTROID]", centroid_wgs84.x.mean(), centroid_wgs84.y.mean())
    print("[BOUNDS WGS84]", aoi_wgs84.total_bounds) 
    aoi = aoi.copy()
    aoi["centroid"] = centroid_wgs84.values
    lon, lat = float(centroid_wgs84.x.mean()), float(centroid_wgs84.y.mean())
    
    epsg, zone, lon_norm = utm_epsg_from_lonlat(lon, lat)
    print(f"[UTM] lon_raw={lon:.6f}, lon_norm={lon_norm:.6f}, lat={lat:.6f} → zone={zone}, {epsg}")
    aoi_utm = aoi_wgs84.to_crs(epsg)
    tz = find_timezone(lon, lat)

    bbox_wgs84 = aoi_wgs84.total_bounds.tolist()
    bbox_utm = aoi_utm.total_bounds.tolist()

    meta = {
        "aoi_wgs84": aoi_wgs84,
        "aoi_utm": aoi_utm,
        "epsg": epsg,
        "tz": tz,
        "lon": lon,
        "lat": lat,
        "bbox_wgs84": bbox_wgs84,
        "bbox_utm": bbox_utm,
        "wind_source": wind_source,
        "Uref": None,          # filled in after Landsat search, see below
        "zref": zref,
        "overpass_time_utc": None,
    }

    with make_gdal_env():
        ls_items = search_sign_items(
            bbox_wgs84, datetime_range, collections=["landsat-c2-l2"], query={"platform": {"in": ["landsat-8", "landsat-9"]},"eo:cloud_cover": {"lt": 2}},
        )
        print(f"Landsat items: {len(ls_items)}")
        required_bands = ["red","green","blue","nir08","swir16","swir22","lwir11"]
        composite = stack_and_composite(ls_items, required_bands, epsg, bbox_utm, res=30, chunksize=2048)
        
        sun_elev = np.median([it.properties["view:sun_elevation"] for it in ls_items])
        sun_azim = np.median([it.properties["view:sun_azimuth"] for it in ls_items])

        # overpass timestamp, same "median across items" pattern as sun_elev/sun_azim above
        ts_ns = np.median([pd.Timestamp(it.properties["datetime"]).value for it in ls_items])
        overpass_time = pd.Timestamp(int(ts_ns))

        if wind_source == "era5":
            try:
                Uref = get_era5_uref(lon, lat, overpass_time, cache_dir=era5_cache_dir)
                print(f"[ERA5] Uref={Uref:.3f} m/s at {overpass_time} (AOI centroid)")
            except Exception as e:
                print(f"[ERA5] fetch failed ({e!r}) -- falling back to fixed Uref={Uref_fixed}")
                Uref = Uref_fixed
        else:
            Uref = Uref_fixed

        meta["Uref"] = Uref
        meta["overpass_time_utc"] = str(overpass_time)

        # DEM
        match_da = composite.sel(band="lwir11")
        bounds_match_utm = list(match_da.rio.bounds())
        bounds_match_utm_padded = pad_bounds(bounds_match_utm, pad_m=30)
        bounds_match_wgs84 = bounds_to_wgs84(bounds_match_utm_padded, epsg)
        dem_items = search_sign_items(bounds_match_wgs84, collections=["cop-dem-glo-30"])
        print(f"DEM items: {len(dem_items)}")
        dem_match = build_dem(dem_items, epsg, bbox_utm=bounds_match_utm_padded, match_da=match_da)
        print("DEM matched")
 
    ds = composite.to_dataset(dim="band").assign(DEM=dem_match)
    ds = ds.rio.clip(aoi_utm.geometry, aoi_utm.crs, drop=True, all_touched=True)
    ds = ds.chunk({"y": 2048, "x": 2048})
    print("array dataset made")
    img1 = add_spectral_indices(ds)
    print("Spectral indices added")
    img2 = compute_radiation(img1, sun_elev, sun_azim)
    print("Radiation terms added")
    img3 = compute_fluxes(img2, Uref=meta["Uref"], zref=meta["zref"])
    print("Flux terms added")
    out = compute_ET(img3, meta["lat"], meta["lon"], tz, datetime_range)
    return out, meta
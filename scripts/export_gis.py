"""Export the rain gauges, their yearly coverage and the rockfalls to a GeoPackage.

For looking at the station situation in QGIS (diagnostic; it doesn't change the
pipeline). Run from the repo root, after a formatter run:

    uv run python -m scripts.export_gis [--run <run_id>]

It reads the outputs of that run (default: the latest; see run_selection.py).
Then build the styled QGIS project with QGIS's own Python:

    /usr/bin/python3.14 -m scripts.build_qgis_project [--run <run_id>]

Output: runs/<run_id>/analyses/gis/riscbal_gauges.gpkg, with these layers
(EPSG:32631):
    gauges            one point per AEMET gauge in the file (all islands)
    gauge_years       one point per gauge and year with data (1940–2022): share
                      of days measured and share of accumulated (-4) days; it
                      has year_start / year_end dates for the Temporal Controller
    api_only_gauges   stations in the AEMET OpenData inventory but not in our file
    rockfalls         the rockfalls of the current export, with their coverage
                      and the nearest gauge that has data for their window
    rockfall_links    lines from each rockfall to its assigned gauge, and, for
                      the under-measured ones, to that nearest gauge with data

Which gauges the API serves comes from AEMET's station inventory, cached in
cache/aemet_api/inventory.csv (downloaded once with AEMET_API_KEY if missing).
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import LineString

from scripts.aemet_api_check import BASE_URL, _get_json
from scripts.run_selection import (
    CACHE_PATH,
    add_run_argument,
    analysis_folder,
    run_folder,
)

DATA_PATH = Path("data")
GPKG_FILE = "riscbal_gauges.gpkg"
AEMET_TABLE_FILE = "formatted_historical_aemet_data.csv.gz"
INVENTORY_FILE = CACHE_PATH / "aemet_api" / "inventory.csv"
CRS = "EPSG:32631"

# Same coverage rule as the controls and the coverage analysis.
MIN_DAYS_90D = 81
WINDOW_DAYS = 90
P_COLUMNS = [f"P{d}" for d in range(1, 32)]

# Some names in the AEMET delivery have lost their accents (U+FFFD in the
# file itself). Display names only; codes are what the pipeline uses.
NAME_FIXES = {
    "B051A": "SÓLLER",
    "B087": "BANYALBUFAR",
    "B158X": "CALVIÀ, ES CAPDELLÀ",
    "B187": "PUIGPUNYENT R. GALATZÓ",
    "B268": "SANTA MARIA SON TORRELLETA",
    "B275C": "MARRATXÍ ES PINARET",
    "B410B": "SANTANYÍ",
    "B512": "ARTÀ CAMPAMENT DES SOLDATS",
    "B526X": "ARTÀ-MOLÍ D'EN LEU",
    "B603X": "COLÒNIA SANT PERE-CAN MENGOL",
    "B656A": "SANTA MARIA DEL CAMÍ",
    "B760X": "POLLENÇA-POLIESPORTIU",
    "B766": "PORT POLLENÇA GOTMAR",
    "B780X": "PORT POLLENÇA-AERÒDROM",
    "B800X": "EVC_MAÓ",
    "B954": "IBIZA/ES CODOLÀ",
}


def _dms(text: str) -> float:
    """AEMET inventory coordinate ('394924N', '025309E') to decimal degrees.

    Parameters
    ----------
    text : str
        DDMMSS or DDDMMSS followed by the hemisphere letter.

    Returns
    -------
    float
        Decimal degrees, negative for W and S.
    """
    hemisphere, digits = text[-1], text[:-1]
    value = (
        int(digits[:-4]) + int(digits[-4:-2]) / 60 + int(digits[-2:]) / 3600
    )
    return -value if hemisphere in "WS" else value


def load_inventory() -> pd.DataFrame:
    """AEMET OpenData station inventory (cached).

    Downloaded once (needs ``AEMET_API_KEY`` in ``.env``) and then read from
    the cache.

    Returns
    -------
    pandas.DataFrame
        One row per station of the API inventory, all columns as text.
    """
    if not INVENTORY_FILE.exists():
        load_dotenv()
        api_key = os.getenv("AEMET_API_KEY")
        if not api_key:
            raise SystemExit(
                "No cached inventory and AEMET_API_KEY is not set in .env"
            )
        answer = _get_json(
            BASE_URL
            + "/valores/climatologicos/inventarioestaciones/todasestaciones",
            api_key,
        )
        INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_get_json(answer["datos"], api_key)).to_csv(
            INVENTORY_FILE, index=False
        )
    return pd.read_csv(INVENTORY_FILE, dtype=str)


def raw_daily() -> pd.DataFrame:
    """The raw AEMET file, one row per gauge-day, keeping the special codes.

    Returns
    -------
    pandas.DataFrame
        ``code``, ``date`` and ``accumulated``: the day is part of a -4 run
        or is the day that closes it (its value is the run's total, spread
        evenly by aemet_formatter.py).
    """
    raw = pd.read_csv(
        DATA_PATH / "historical_aemet_data.csv", sep=";", dtype=str
    )
    daily = raw.melt(
        id_vars=["INDICATIVO", "AÑO", "MES"],
        value_vars=P_COLUMNS,
        var_name="day",
        value_name="value",
    )
    daily["date"] = pd.to_datetime(
        {
            "year": daily["AÑO"].astype(int),
            "month": daily["MES"].astype(int),
            "day": daily["day"].str[1:].astype(int),
        },
        errors="coerce",
    )
    daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
    daily = daily.dropna(subset=["date", "value"]).sort_values(
        ["INDICATIVO", "date"]
    )
    previous = daily.groupby("INDICATIVO")["value"].shift(1)
    daily["accumulated"] = (daily["value"] == -4) | (
        (previous == -4) & (daily["value"] != -4)
    )
    return daily.rename(columns={"INDICATIVO": "code"})[
        ["code", "date", "accumulated"]
    ]


def main() -> None:
    """Write ``runs/<run_id>/analyses/gis/riscbal_gauges.gpkg``.

    See the module docstring for the layers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_argument(parser)
    run = run_folder(parser.parse_args().run)
    outputs = run / "outputs"
    out_file = analysis_folder(run, "gis") / GPKG_FILE
    formatted = pd.read_csv(outputs / AEMET_TABLE_FILE)
    formatted["date"] = pd.to_datetime(formatted["date"], format="%Y-%m-%d")
    measured = formatted[formatted["precipitation"].notna()]

    inventory = load_inventory()
    stations_csv = pd.read_csv(DATA_PATH / "stations.csv")
    final = pd.read_csv(outputs / "final_gdf.csv")
    sample_gauges = pd.read_csv(outputs / "samples_with_stations.csv")[
        ["sample_id", "code"]
    ]
    used_codes = set(
        stations_csv[stations_csv["id"].isin(final["id_station"])]["code"]
    )

    # --- gauges ---
    gauges = formatted.groupby("code").agg(
        name=("NOMBRE", "first"),
        altitude_m=("ALTITUD", "first"),
        lon=("lon", "first"),
        lat=("lat", "first"),
    )
    first_last = measured.groupby("code")["date"].agg(["min", "max"])
    gauges["first_data"] = first_last["min"].dt.strftime("%Y-%m-%d")
    gauges["last_data"] = first_last["max"].dt.strftime("%Y-%m-%d")
    gauges["name"] = [
        NAME_FIXES.get(code, " ".join(str(n).split()))
        for code, n in gauges["name"].items()
    ]
    gauges["in_api"] = gauges.index.isin(inventory["indicativo"])
    gauges["gauge_type"] = np.where(gauges["in_api"], "automatic", "manual")
    gauges["in_stations_csv"] = gauges.index.isin(stations_csv["code"])
    gauges["used_by_polygons"] = gauges.index.isin(used_codes)
    gauges = (
        gpd.GeoDataFrame(
            gauges.reset_index(),
            geometry=gpd.points_from_xy(gauges["lon"], gauges["lat"]),
            crs="EPSG:4326",
        )
        .to_crs(CRS)
        .drop(columns=["lon", "lat"])
    )

    # --- gauge_years: share of days measured and accumulated, per year ---
    daily = raw_daily()
    daily["year"] = daily["date"].dt.year
    per_year = (
        daily.groupby(["code", "year"])
        .agg(days=("date", "size"), accumulated_days=("accumulated", "sum"))
        .reset_index()
    )
    days_in_year = np.where(
        pd.Series(per_year["year"]).map(
            lambda y: pd.Timestamp(y, 12, 31).dayofyear
        )
        == 366,
        366,
        365,
    )
    per_year["coverage"] = (per_year["days"] / days_in_year).round(3)
    per_year["accumulated_share"] = (
        per_year["accumulated_days"] / days_in_year
    ).round(3)
    per_year["year_start"] = pd.to_datetime(
        per_year["year"].astype(str) + "-01-01"
    )
    per_year["year_end"] = pd.to_datetime(
        per_year["year"].astype(str) + "-12-31"
    )
    gauge_years = gauges[
        ["code", "name", "gauge_type", "in_stations_csv", "geometry"]
    ].merge(per_year, on="code")
    gauge_years["coverage_class"] = np.select(
        [gauge_years["coverage"] >= 0.9, gauge_years["coverage"] > 0],
        ["≥ 90% of days", "some days"],
        "none",
    )

    # --- API stations missing from our file ---
    balearic = inventory[
        inventory["provincia"].str.contains("BALEAR", na=False)
    ]
    api_only = balearic[~balearic["indicativo"].isin(gauges["code"])]
    api_only = gpd.GeoDataFrame(
        {
            "code": api_only["indicativo"],
            "name": api_only["nombre"],
            "altitude_m": api_only["altitud"].astype(int),
        },
        geometry=gpd.points_from_xy(
            api_only["longitud"].map(_dms), api_only["latitud"].map(_dms)
        ),
        crs="EPSG:4326",
    ).to_crs(CRS)

    # --- rockfalls and the nearest gauge with data for their window ---
    rockfalls = final[final["sample_type"] == "rockfall"].merge(
        sample_gauges, on="sample_id", how="left"
    )
    rockfalls["date"] = pd.to_datetime(rockfalls["date"], format="%Y-%m-%d")
    measured_days = measured.groupby("code")["date"].apply(set)
    gauge_xy = gauges.set_index("code").geometry

    def nearest_with_data(row: pd.Series) -> tuple[str | None, float | None]:
        """Nearest gauge (any distance) that covers a rockfall's window.

        Parameters
        ----------
        row : pandas.Series
            One rockfall, with ``date``, ``POINT_X`` and ``POINT_Y``.

        Returns
        -------
        code : str or None
            The gauge, or None if no gauge covers the window.
        distance_km : float or None
            Its distance to the rockfall, in km (0.1 km).
        """
        event_day = row["date"]
        window = [
            event_day - pd.Timedelta(days=k) for k in range(1, WINDOW_DAYS + 1)
        ]
        distances = gauge_xy.distance(
            gpd.points_from_xy([row["POINT_X"]], [row["POINT_Y"]])[0]
        ).sort_values()
        for code, metres in distances.items():
            days = measured_days.get(code, set())
            if (
                event_day in days
                and sum(d in days for d in window) >= MIN_DAYS_90D
            ):
                return code, round(metres / 1000, 1)
        return None, None

    rockfalls["under_measured"] = rockfalls["rain_days_90d"] < MIN_DAYS_90D
    nearest = [
        nearest_with_data(r) if r["under_measured"] else (None, None)
        for _, r in rockfalls.iterrows()
    ]
    rockfalls["nearest_gauge_with_data"] = [n[0] for n in nearest]
    rockfalls["nearest_km"] = [n[1] for n in nearest]
    rockfalls["assigned_km"] = [
        round(gauge_xy[c].distance(gpd.points_from_xy([x], [y])[0]) / 1000, 1)
        if c in gauge_xy.index
        else None
        for c, x, y in zip(
            rockfalls["code"], rockfalls["POINT_X"], rockfalls["POINT_Y"]
        )
    ]
    rockfall_layer = gpd.GeoDataFrame(
        rockfalls[
            [
                "sample_id",
                "municipality",
                "date",
                "code",
                "assigned_km",
                "rain_days_90d",
                "under_measured",
                "event_day_acc",
                "last_week_acc",
                "last_month_acc",
                "last_three_months_acc",
                "nearest_gauge_with_data",
                "nearest_km",
            ]
        ].rename(columns={"code": "assigned_gauge"}),
        geometry=gpd.points_from_xy(
            rockfalls["POINT_X"], rockfalls["POINT_Y"]
        ),
        crs=CRS,
    )
    rockfall_layer["year"] = rockfall_layer["date"].dt.year
    rockfall_layer["date"] = rockfall_layer["date"].dt.strftime("%Y-%m-%d")

    # --- links ---
    links = []
    for _, r in rockfall_layer.iterrows():
        for kind, code in (
            ("assigned", r["assigned_gauge"]),
            ("nearest with data", r["nearest_gauge_with_data"]),
        ):
            if code in gauge_xy.index:
                links.append(
                    {
                        "sample_id": r["sample_id"],
                        "link": kind,
                        "gauge": code,
                        "under_measured": r["under_measured"],
                        "geometry": LineString([r.geometry, gauge_xy[code]]),
                    }
                )
    links = gpd.GeoDataFrame(links, crs=CRS)

    out_file.unlink(missing_ok=True)
    for name, layer in (
        ("gauges", gauges),
        ("gauge_years", gauge_years),
        ("api_only_gauges", api_only),
        ("rockfalls", rockfall_layer),
        ("rockfall_links", links),
    ):
        layer.to_file(out_file, layer=name, driver="GPKG")
        print(f"{name}: {len(layer)} features")
    print(f"Written to {out_file}")


if __name__ == "__main__":
    main()

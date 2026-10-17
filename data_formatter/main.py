import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from dotenv import load_dotenv

from data_formatter.aemet_formatter import format_aemet_data
from data_formatter.eda import dataframe_analysis
from data_formatter.inventory import SAMPLES_DATE_FORMAT, build_samples
from data_formatter.prec_accumulation import get_prec_accumulations_by_date

load_dotenv()

DATA_PATH = Path("./data")
RESULTS_PATH = Path("./results")

# --------------------------------------------------------------------------
# Negative sampling: shift some negatives to high-rainfall days
# --------------------------------------------------------------------------
# Original design: move SHIFT_SHARE of the negatives to a random day with
# rain >= the station's SHIFT_PERCENTILE (within SHIFT_PERIOD), so the model
# can't separate the classes on "it rained" alone.
#
# Phase B0 (2026-10-17): switched OFF. With the shift, the moved negatives
# are much wetter on the event day than the rockfalls (mean 30.6 vs 8.8 mm),
# and every model learned "more rain -> less risk". Without them, all four
# rain windows correlate positively with rockfalls. This run tests whether
# the SHAP sign of event_day_acc flips in the models too. It is a diagnostic,
# not the final design: without the shift almost no negative has rain on its
# day (see docs/08-limitations.md, and Phase B1 in TODO.md).
SHIFT_NEGATIVES_TO_RAINY_DAYS = False
SHIFT_SHARE = 0.4
SHIFT_PERCENTILE = 0.95
SHIFT_PERIOD = ("2005-01-01", "2015-12-31")
SHIFT_SEED = 42

# 1. Cargar los CSV originales
df_stations = pd.read_csv(
    DATA_PATH / 'stations.csv',
)

# df_stations.to_csv(RESULTS_PATH / 'stations.csv', index=True, index_label="id")

# Samples in samples.csv format: rockfalls from the original inventory
# (correct day-first dates) and recent_rockfalls.csv, negatives from
# negatives.csv. samples.csv is no longer read. See data_formatter/inventory.py.
df_samples = build_samples(DATA_PATH, RESULTS_PATH)
#
# df_samples = df_samples.drop(columns=[
#    "OBJECTID_1", "Tipo", "Tipo_code", "Material",
#    "Daños", "GEO_SIMPLI", "GEO_SIMPL_code",
#    "FID_1", "Id", "OBJECTID", "localizaci",
#    "RegistroD", "RegistroR", "Año_1", "Municipio_1",
#    "Vertiente_1", "H25", "Tipo_1", "Ciclos_hie",
#    "Material_1", "Daños_1", "Valoración", "Precipitac",
#    "Ocurrencia", "Registro_f", "Fuente_dat", "Mes_1",
#    "Fecha_entr", "classifica", "Damage", "Size", "Volume",
#    "Dir_Cost", "Ind_Cost", "TIPOLOGIA", "Temporada"
#
# ])

# df_samples.to_csv(DATA_PATH / 'samples.csv', index=False)

# 2. Convertir los DataFrame normales a GeoDataFrames (Capas espaciales)
# Asumimos que tus columnas se llaman 'lon' (longitud) y 'lat' (latitud)
# El CRS (Coordinate Reference System) EPSG:4326 es el estándar GPS (WGS84)

#gdf_stations = gpd.GeoDataFrame(
#    df_stations,
#    geometry=gpd.points_from_xy(df_stations['utm_x'], df_stations['utm_y']),
#    crs="EPSG:32631"
#)
#
#gdf_samples = gpd.GeoDataFrame(
#    df_samples,
#    geometry=gpd.points_from_xy(df_samples['POINT_X'], df_samples['POINT_Y']),
#    crs="EPSG:32631"
#)

# (Opcional) Si realmente necesitas guardar el Shapefile de las estaciones
# para verlo en otro programa, puedes descomentar la siguiente línea:
# gdf_estaciones.to_file("capa_estaciones.shp")

# ==========================================
# Join nearest stations to sample
# ==========================================

# Cruzamos las muestras con las estaciones.
# sjoin_nearest asigna cada muestra a la estación más cercana (lógica de Thiessen)
#samples_with_station = gpd.sjoin_nearest(
#    left_df=gdf_samples,
#    right_df=gdf_stations,
#    how="left",
#)
samples_with_station_df = pd.merge(
    df_samples,
    df_stations,
    left_on="id_station",
    right_on="id",
    how="inner"
)

samples_with_station_df = samples_with_station_df.drop(columns=[
    'utm_x', 'utm_y', "id"
])

os.makedirs(RESULTS_PATH, exist_ok=True)

samples_with_station_df = samples_with_station_df.rename(columns={
    "Fecha": "date",
    "Tipus": "is_rockfall",
    "Municipio": "municipality",
    "Vertiente": "hillside",
    "DURESA": "hardness"
})
# Explicit format: a date in any other format fails loudly instead of being
# silently re-interpreted (an ambiguous 8/1/2009 would become August 1st).
samples_with_station_df["date"] = pd.to_datetime(samples_with_station_df["date"], format=SAMPLES_DATE_FORMAT)
samples_with_station_df.loc[
    (samples_with_station_df["municipality"] == "Esporles")
    & (samples_with_station_df["date"] > "2024-03-25")
    & (samples_with_station_df["date"] <= "2025-03-25"), "code"] = "GRO08"
samples_with_station_df.loc[
    (samples_with_station_df["municipality"] == "Esporles")
    & (samples_with_station_df["date"] == "2026-01-21"), "code"] = "BAR04"

samples_with_station_df['hardness'] = samples_with_station_df['hardness'].replace({'Moderately': 'Moderately hard rocks'})

# Guardamos el resultado listo para tu modelo
samples_with_station_df.to_csv(RESULTS_PATH / 'samples_with_stations.csv', index=False)

aemet_df = pd.read_csv(DATA_PATH / "historical_aemet_data.csv", sep=";")
aemet_df = format_aemet_data(aemet_df)

aemet_df.to_csv(RESULTS_PATH / "formatted_historical_aemet_data.csv", index=False)

# One row per negative actually moved: where, which station, old and new date.
shifted_negatives = []
if SHIFT_NEGATIVES_TO_RAINY_DAYS:
    percentiles = aemet_df.groupby('code')['precipitation'].quantile(SHIFT_PERCENTILE)

    # Get high-precipitation days (>= station percentile) within the period
    aemet_df_filtered = aemet_df[
        (aemet_df["date"] >= SHIFT_PERIOD[0]) & (aemet_df["date"] <= SHIFT_PERIOD[1])
    ].copy()

    high_precipitation_days = aemet_df_filtered[
        aemet_df_filtered["precipitation"] >= aemet_df_filtered["code"].map(percentiles)
    ]

    high_precipitation_days_by_station = (
        high_precipitation_days.groupby("code")["date"].apply(lambda dates: dates.tolist()).to_dict()
    )

    # Replace the date for SHIFT_SHARE of is_rockfall == 'F' samples with a
    # random high-precipitation day belonging to the same station code.
    rng = np.random.default_rng(SHIFT_SEED)
    f_mask = samples_with_station_df["is_rockfall"] == "F"
    f_indices = samples_with_station_df.loc[f_mask].index
    n_replace = int(len(f_indices) * SHIFT_SHARE)
    replace_indices = rng.choice(f_indices, size=n_replace, replace=False)

    for idx in replace_indices:
        station_code = samples_with_station_df.at[idx, "code"]
        available_dates = high_precipitation_days_by_station.get(station_code, [])
        if available_dates:
            old_date = samples_with_station_df.at[idx, "date"]
            samples_with_station_df.at[idx, "date"] = rng.choice(available_dates)
            shifted_negatives.append({
                "POINT_X": samples_with_station_df.at[idx, "POINT_X"],
                "POINT_Y": samples_with_station_df.at[idx, "POINT_Y"],
                "code": station_code,
                "original_date": old_date,
                "shifted_date": samples_with_station_df.at[idx, "date"],
            })
pd.DataFrame(
    shifted_negatives, columns=["POINT_X", "POINT_Y", "code", "original_date", "shifted_date"]
).to_csv(RESULTS_PATH / "shifted_negatives.csv", index=False)
print(f"Negatives shifted to high-rainfall days: {len(shifted_negatives)}"
      f" (shift {'ON' if SHIFT_NEGATIVES_TO_RAINY_DAYS else 'OFF'})")

samples_with_station_df["event_day_acc"] = 0.0
samples_with_station_df["last_week_acc"] = 0.0
samples_with_station_df["last_month_acc"] = 0.0
samples_with_station_df["last_three_months_acc"] = 0.0
final_df = samples_with_station_df.apply(lambda row: get_prec_accumulations_by_date(row, aemet_df), axis=1)
final_df.to_csv(RESULTS_PATH / "samples_with_accumulations.csv", index=False)

susceptibility_df = gpd.read_file(DATA_PATH / "susceptibility" / "vectorial_gravitacionals_INES.shp")
final_gdf = gpd.GeoDataFrame(
    final_df,
    geometry=gpd.points_from_xy(final_df['POINT_X'], final_df['POINT_Y']),
    crs="EPSG:32631"
)

susceptibility_df = susceptibility_df.to_crs(final_gdf.crs)

final_gdf = gpd.sjoin(
    left_df=final_gdf,
    right_df=susceptibility_df,
    how="left",
    predicate="within"
)

final_gdf = final_gdf.drop(columns=["index_right", "geometry", "fid", "Value", "code"])
final_gdf = final_gdf.rename(columns={"POTENCIALI": "susceptibility"})
final_gdf.to_csv(RESULTS_PATH / "samples_with_susceptibility.csv", index=False)

final_gdf['is_rockfall'] = final_gdf['is_rockfall'].map({"T": 1, "F": 0})

dataframe_analysis(final_gdf)

columns_with_slope = list(final_gdf.columns[final_gdf.columns.str.contains("slope")])
columns_with_elevation = list(final_gdf.columns[final_gdf.columns.str.contains("elev")])

final_gdf = final_gdf.drop(columns=columns_with_slope + columns_with_elevation)
final_gdf.to_csv(RESULTS_PATH / "final_gdf.csv", index=False)


# Record how this export was made. final_gdf_sha256 is the same hash the
# modelling repo stores as data_sha256 in each experiment's run_config.json,
# so an experiment can be traced back to these settings.
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


formatter_config = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "negative_shift": {
        "enabled": SHIFT_NEGATIVES_TO_RAINY_DAYS,
        "share": SHIFT_SHARE,
        "percentile": SHIFT_PERCENTILE,
        "period": list(SHIFT_PERIOD),
        "seed": SHIFT_SEED,
        "n_shifted": len(shifted_negatives),
    },
    "n_samples": len(final_gdf),
    "n_positives": int(final_gdf["is_rockfall"].sum()),
    "inputs_sha256": {
        path.name: _sha256(path)
        for path in sorted(DATA_PATH.glob("*.csv"))
    },
    "final_gdf_sha256": _sha256(RESULTS_PATH / "final_gdf.csv"),
}
(RESULTS_PATH / "formatter_config.json").write_text(json.dumps(formatter_config, indent=2) + "\n")

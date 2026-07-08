import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from aemet_formatter import format_aemet_data
from data_formatter.prec_accumulation import get_prec_accumulations_by_date

load_dotenv()

DATA_PATH = Path("./data")
RESULTS_PATH = Path("./results")

# 1. Cargar los CSV originales
df_stations = pd.read_csv(
    DATA_PATH / 'stations.csv',
)

# df_stations.to_csv(RESULTS_PATH / 'stations.csv', index=True, index_label="id")

df_samples = pd.read_csv(DATA_PATH / 'samples.csv', sep=';')
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
    'utm_x', 'utm_y', "id_station", "id"
])

os.makedirs(RESULTS_PATH, exist_ok=True)

samples_with_station_df = samples_with_station_df.rename(columns={
    "Fecha": "date",
    "Tipus": "is_rockfall",
    "Municipio": "municipality",
    "Vertiente": "slope",
    "DURESA": "hardness"
})
samples_with_station_df["date"] = pd.to_datetime(samples_with_station_df["date"])
samples_with_station_df.loc[
    (samples_with_station_df["municipality"] == "Esporles")
    & (samples_with_station_df["date"] > "2024-03-25")
    & (samples_with_station_df["date"] <= "2025-03-25"), "code"] = "GRO08"
samples_with_station_df.loc[
    (samples_with_station_df["municipality"] == "Esporles")
    & (samples_with_station_df["date"] == "2026-01-21"), "code"] = "BAR04"

# Guardamos el resultado listo para tu modelo
samples_with_station_df.to_csv(RESULTS_PATH / 'samples_with_stations.csv', index=False)

aemet_df = pd.read_csv(DATA_PATH / "historical_aemet_data.csv", sep=";")
aemet_df = format_aemet_data(aemet_df)

aemet_df.to_csv(RESULTS_PATH / "formatted_historical_aemet_data.csv", index=False)

samples_with_station_df["last_day_acc"] = 0.0
samples_with_station_df["last_week_acc"] = 0.0
samples_with_station_df["last_month_acc"] = 0.0
samples_with_station_df["last_three_months_acc"] = 0.0
final_df = samples_with_station_df.apply(lambda row: get_prec_accumulations_by_date(row, aemet_df), axis=1)
final_df = final_df.rename(columns={"code": "station_code"})
final_df.to_csv(RESULTS_PATH / "samples_with_accumulations.csv", index=False)

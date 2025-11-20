import pandas as pd
import kagglehub

from data.download_kaggle import download_unicon

# Pobierz dane (wykonaj raz)
datasets = download_unicon()

# Dostęp do poszczególnych dataframe'ów
df_weather = datasets['weather']
df_bc = datasets['consumption']
df_events = datasets['events']
df_cal = datasets['calendar']
df_building_meta = datasets['building_meta']
df_campus_meta = datasets['campus_meta']








# Download latest version
path = kagglehub.dataset_download("cdaclab/unicon")

print("Path to dataset files:", path)



df_weather = pd.read_csv(f"{path}\weather_data.csv")
df_bc = pd.read_csv(f"{path}/building_consumption.csv")
df_events = pd.read_csv(f"{path}/events.csv")
df_cal = pd.read_csv(f"{path}/calender.csv")
#%%
datasets = [df_weather, df_bc, df_events, df_cal]
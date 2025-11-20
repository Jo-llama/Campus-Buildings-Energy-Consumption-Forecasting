
def preprocess_and_merge(df_weather, df_bc, df_events, df_cal):
    # Konwersja dat
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_bc['timestamp'] = pd.to_datetime(df_bc['timestamp'])
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_cal['date'] = pd.to_datetime(df_cal['date'])

    # Agregacja do poziomu dni
    df_bc['date'] = df_bc['timestamp'].dt.date
    df_weather['date'] = df_weather['timestamp'].dt.date

    # Grupowanie zużycia po dniach
    df_bc_daily = df_bc.groupby(['campus_id', 'meter_id', 'date'], as_index=False)['consumption'].sum()

    # Agregacja pogodowa po dniach i kampusach
    df_weather_daily = df_weather.groupby(['campus_id', 'date'], as_index=False).agg({
        'air_temperature':'mean',
        'apparent_temperature':'mean',
        'dew_point_temperature':'mean',
        'relative_humidity':'mean',
        'wind_speed':'mean',
        'wind_direction':'mean'
    })

    # Łączenie zużycia i pogody
    df = pd.merge(df_bc_daily, df_weather_daily, on=['campus_id','date'], how='left')

    # Dołączenie danych kalendarza
    df = pd.merge(df, df_cal, on='date', how='left')

    # Eventy: stworzenie cechy "days_since_last_event"
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_events = df_events.sort_values(by=['meter_id', 'date'])

    last_event_date = df_events.groupby('meter_id')['date'].max().reset_index().rename(columns={'date':'last_event_date'})
    df = pd.merge(df, last_event_date, on='meter_id', how='left')

    df['days_since_last_event'] = (pd.to_datetime(df['date']) - df['last_event_date']).dt.days
    df['days_since_last_event'].fillna(9999, inplace=True)

    return df

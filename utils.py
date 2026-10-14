
def preprocess_and_merge(df_weather, df_bc, df_events, df_cal):
    # Convert dates
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_bc['timestamp'] = pd.to_datetime(df_bc['timestamp'])
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_cal['date'] = pd.to_datetime(df_cal['date'])

    # Aggregate to daily level
    df_bc['date'] = df_bc['timestamp'].dt.date
    df_weather['date'] = df_weather['timestamp'].dt.date

    # Group consumption by day
    df_bc_daily = df_bc.groupby(['campus_id', 'meter_id', 'date'], as_index=False)['consumption'].sum()

    # Aggregate weather by day and campus
    df_weather_daily = df_weather.groupby(['campus_id', 'date'], as_index=False).agg({
        'air_temperature':'mean',
        'apparent_temperature':'mean',
        'dew_point_temperature':'mean',
        'relative_humidity':'mean',
        'wind_speed':'mean',
        'wind_direction':'mean'
    })

    # Join consumption and weather
    df = pd.merge(df_bc_daily, df_weather_daily, on=['campus_id','date'], how='left')

    # Join calendar data
    df = pd.merge(df, df_cal, on='date', how='left')

    # Events: create the "days_since_last_event" feature
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_events = df_events.sort_values(by=['meter_id', 'date'])

    last_event_date = df_events.groupby('meter_id')['date'].max().reset_index().rename(columns={'date':'last_event_date'})
    df = pd.merge(df, last_event_date, on='meter_id', how='left')

    df['days_since_last_event'] = (pd.to_datetime(df['date']) - df['last_event_date']).dt.days
    df['days_since_last_event'].fillna(9999, inplace=True)

    return df

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from tqdm import tqdm
from src.utils.functions import mkdir_if_not_exists, convert_time_to_float
from src.utils.logger import log
from logging import INFO
warnings.filterwarnings("ignore")



class ParticipantData:
    @classmethod
    def catch_data(cls, _id: int, path: str = f'data/pecanstreet/aggregate/'):
        return {"_id": _id, "data": pd.read_csv(f"{path}/{str(_id)}.csv")}

    @staticmethod
    def init_weather_readings(path: str = 'data/pecanstreet/') -> pd.DataFrame:
        try:
            weather_df = pd.read_csv(f"{path}/weather_data/162.89.0.47.csv")
        except:
            raise FileExistsError(
                '[!] - Please, make sure that you have the weather features available for the specific location!')

        weather_df['date'] = pd.to_datetime(weather_df['date_time'])
        del weather_df['moonrise'], weather_df['moonset'], weather_df['sunrise'], weather_df['sunset']

        weather = []
        for _, row in tqdm(weather_df.iterrows(), total=weather_df.shape[0]):
            values = {
                'date': datetime.strftime(row.date, '%Y-%m-%d'),
                'hour': datetime.strftime(row.date, '%H:%M')
            }
            for columns in weather_df.columns[1:-1]:
                values[columns] = row[columns]
            weather.append(values)

        weather_df = pd.DataFrame(weather)
        return weather_df

    @classmethod
    def preprocess_readings(cls, data_path: str = 'data/pecanstreet/', location: str='austin'):
        df = pd.read_csv(data_path, parse_dates=[1])
        df['local_15min'] = pd.to_datetime(df['local_15min'], utc=True)
        df = df.sort_values(by='local_15min').reset_index(drop=True)
        cid = df['dataid'].unique()[0]
        log(INFO, f"Preprocessing readings from {cid}")
        #
        df['solar'].fillna(0, inplace=True)
        df['solar2'].fillna(0, inplace=True)
        df['generation'] = df['solar'] + df['solar2']
        df['generation'].clip(lower=0, inplace=True)
        cols = [col for col in df.columns if col not in ['dataid', 'local_15min', 'grid', 'leg1v', 'leg2v', 'generation']]
        df.drop(cols, axis=1, inplace=True)
        # consumption_cols = df.columns[2:len(df.columns) - 2]
        df['consumption'] = df['grid']
        df.drop(['grid'], axis=1, inplace=True)
        df['year'] = df['local_15min'].dt.year
        df['month'] = df['local_15min'].dt.month
        df['day'] = df['local_15min'].dt.day
        df['hour'] = df['local_15min'].dt.hour
        df['minute'] = df['local_15min'].dt.minute
        df['second'] = df['local_15min'].dt.second
        df['day_of_week'] = df['local_15min'].dt.dayofweek
        df['week_of_year'] = df['local_15min'].dt.isocalendar().week
        df['prev_consumption'] = df.shift(1)['consumption']
        df['consumption_change'] = df.apply(
            lambda row: 0 if np.isnan(row.prev_consumption) else row.consumption - row.prev_consumption, axis=1)

        # df['prev_generation_solar1'] = df.shift(1)['generation_solar1']
        # df['change_generation_solar1'] = df.apply(
        #     lambda row: 0 if np.isnan(row['prev_generation_solar1']) else row['generation_solar1'] - row[
        #         'prev_generation_solar1'], axis=1)
        #
        # df['prev_generation_solar2'] = df.shift(1)['generation_solar2']
        # df['change_generation_solar2'] = df.apply(
        #     lambda row: 0 if np.isnan(row['prev_generation_solar2']) else row['generation_solar2'] - row[
        #         'prev_generation_solar2'], axis=1)
        df.drop('dataid', inplace=True, axis=1)
        df.rename(columns={'local_15min': 'Date'}, inplace=True)

        df = cls.insert_weather_data(df, location)
        df['cid'] = cid
        log(INFO, f"{cid} data loaded with shape: {df.shape}")
        return df, cid

    @staticmethod
    def insert_weather_data(df, location):
        def get_data(date):
            temp_ = weather_df.loc[(weather_df['Date'] == date)]
            temp_.reset_index(inplace=True, drop=True)
            temp_.drop('Date', axis=1, inplace=True)
            return json.loads(temp_.to_json(orient='records'))

        data = df.copy()
        try:
            weather_df = pd.read_csv(f'dataset/pecanstreet/weather_data/open-meteo-{location}.csv')
        except:
            raise FileExistsError(
                '[!] - Please, make sure that you have the weather features available for the specific location!')
        finally:
            weather_df['date_time'] = pd.to_datetime(weather_df['time'], utc=True)
            weather_df.drop(['time'], axis=1, inplace=True)
            weather_df.sort_values(by='date_time', inplace=True)
            cols = weather_df.columns.to_list()
            cols = cols[-1:] + cols[:-1]
            weather_df = weather_df[cols]
            weather_df = weather_df.resample('15Min', on='date_time').mean()
            weather_df.interpolate(method='polynomial', order=2, inplace=True)
            weather_df['Date'] = weather_df.index
            weather_df.reset_index(drop=True, inplace=True)
            weather_df = weather_df[weather_df.columns[::-1]]
            final_df = []
            for i, row in tqdm(data.iterrows(), total=data.shape[0]):
                try:
                    row_ = json.loads(row.to_json())
                    temp_ = get_data(row["Date"])
                    if temp_ is None or len(temp_) == 0:
                        temp_ = get_data(pd.Timestamp(f'2018-{row.month}-{row.day} {row.hour}:00:00+0000'))
                    row_.update(temp_[0])
                    final_df.append(row_)
                except:
                    raise Exception
            df_ = pd.DataFrame(final_df)
            df_['Date'] = data['Date']
            return df_

    @classmethod
    def aggregate_features(cls, _id: int, path: str = 'data/pecanstreet/'):
        weather_df = cls.init_weather_readings(path)

        features_df, _id = cls.preprocess_readings(data_path=f"{path}/15min/{str(_id)}.str", weather_df=weather_df)
        mkdir_if_not_exists(f"{path}/aggregate/")
        mkdir_if_not_exists(f"{path}/aggregate/15min")
        del features_df['date'], features_df['hour']
        features_df.to_csv(f"{path}/aggregate/15min/{_id}.csv",index=False)
        return {"_id": _id, "data": features_df}



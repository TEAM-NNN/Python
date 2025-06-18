import azure.functions as func
import logging
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import jpholiday
from collections import defaultdict, Counter
import json


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


OPENWEATHERMAP_API_KEY = "e4e9bafa72e8c8f58e775b0e28f9a875" 
LOCATION = "Kachidoki,jp"

# 天気OpenWeatherMapから受け取る関数
def get_weather_forecast():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={LOCATION}&appid={OPENWEATHERMAP_API_KEY}&lang=ja&units=metric"
    response = requests.get(url)
    data = response.json()

    daily_data = defaultdict(lambda: {
        "temps": [],
        "humidity": [],
        "pressure": [],
        "wind": [],
        "rain": [],
        "weathers": []
    })

    for item in data["list"]:
        dt_utc = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
        dt_jst = dt_utc + timedelta(hours=9)
        date = dt_jst.date()

        daily_data[date]["temps"].append(item["main"]["temp"])
        daily_data[date]["humidity"].append(item["main"]["humidity"])
        daily_data[date]["pressure"].append(item["main"]["pressure"])
        daily_data[date]["wind"].append(item["wind"]["speed"])
        daily_data[date]["weathers"].append(item["weather"][0]["main"])

        rain_volume = item.get("rain", {}).get("3h", 0)
        daily_data[date]["rain"].append(rain_volume)

    records = []
    for date, values in daily_data.items():
        record = {
            "date": date,
            "最高気温": max(values["temps"]),
            "最低気温": min(values["temps"]),
            "平均気温": sum(values["temps"]) / len(values["temps"]),
            "降水量の合計": sum(values["rain"]),
            "平均風速": sum(values["wind"]) / len(values["wind"]),
            "平均気圧": sum(values["pressure"]) / len(values["pressure"]),
            "平均湿度": sum(values["humidity"]) / len(values["humidity"]),
            "has_晴": int("Clear" in values["weathers"]),
            "has_雲": int("Clouds" in values["weathers"]),
            "has_雷": int("Thunderstorm" in values["weathers"]),
        }
        records.append(record)

    return pd.DataFrame(records)

# 天気情報以外の特徴量を追加
def add_features(df):
    df["is_holiday"] = df["date"].apply(lambda x: x in jpholiday.HOLIDAYS)
    df['is_holiday'] = df['is_holiday'].astype(int)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["weekday"] = df["date"].apply(lambda x: x.weekday())
    return df

# 前処理
def preprocess(df, scaler):
    # カテゴリ変数をOneHotエンコード
    df = pd.get_dummies(df, columns=['month', 'weekday'], prefix=['month', 'weekday'], drop_first=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    X = scaler.transform(df.drop(columns=["date"]))  # "date"列は予測に使わない
    return X

# APIエンドポイント
@app.route(route="BeerPredictAPI")
def BeerPredictAPI(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Predicting beer demand for all beer types.')

    try:
        beer_types = ["PALE", "LAGER", "IPA", "WHITE", "BLACK", "FRUIT"]
        model_dir = "models"

        # ビールごとの特徴量定義
        feature_map = {
            "PALE": ['weekday_4', 'has_晴', 'month_11', '最高気温', '降水量の合計', 'has_雷', 'weekday_2', '平均湿度', '平均風速'],
            "LAGER": ['weekday_4', '最高気温', 'has_晴', 'month_7', 'month_9', 'month_8', '降水量の合計', 'weekday_2', 'month_3'],
            "IPA": ['weekday_4', '最高気温', 'has_晴', 'month_9', 'month_7', '降水量の合計', 'has_雷', 'weekday_2', 'month_3'],
            "WHITE": ['最高気温', 'weekday_4', 'month_7', 'month_9', 'has_晴', '降水量の合計', 'weekday_2', 'month_3', 'month_2'],
            "BLACK": ['weekday_4', 'has_晴', 'month_12', 'month_11', '最低気温', '降水量の合計', '平均湿度', 'has_雷', 'weekday_2', '平均風速'],
            "FRUIT": ['weekday_4', '最高気温', 'has_晴', 'month_9', '降水量の合計']
        }

        # 天気情報と追加特徴量を取得
        df = get_weather_forecast()
        df = add_features(df)

        # One-hot エンコード（月・曜日）
        df_encoded = pd.get_dummies(df, columns=["month", "weekday"], prefix=["month", "weekday"], drop_first=True)

        # すべてfloat型に変換
        for col in df_encoded.columns:
            if col != "date":
                df_encoded[col] = df_encoded[col].astype(float)

        all_predictions = {}

        for beer in beer_types:
            model_path = f"{model_dir}/{beer}_best_model.pkl"
            features = feature_map[beer]

            try:
                data = joblib.load(model_path)
                model = data["model"]
                scaler = data["scaler"]

                # 特徴量がそろっているか確認（なければエラー）
                missing = [col for col in features if col not in df_encoded.columns]
                if missing:
                    raise ValueError(f"欠損している特徴量: {missing}")

                X = df_encoded[features]
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)

                all_predictions[beer] = {
                    str(date): round(pred, 2)
                    for date, pred in zip(df["date"], preds)
                }

            except Exception as e:
                logging.warning(f"Could not predict for {beer}: {e}")
                all_predictions[beer] = "Prediction failed"

        return func.HttpResponse(
            body=json.dumps(all_predictions, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Fatal error occurred: {e}")
        return func.HttpResponse(
            f"Internal error: {e}",
            status_code=500
        )
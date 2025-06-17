import azure.functions as func
import logging
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import japan_holidays
from collections import defaultdict, Counter

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# モデルとスケーラーのパス（例：PALEビール用）
OPENWEATHERMAP_API_KEY = "e4e9bafa72e8c8f58e775b0e28f9a875"  # ←ここにAPIキーを設定
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
            "temp_max": max(values["temps"]),
            "temp_min": min(values["temps"]),
            "temp_mean": sum(values["temps"]) / len(values["temps"]),
            "precip_sum": sum(values["rain"]),
            "wind_mean": sum(values["wind"]) / len(values["wind"]),
            "pressure_mean": sum(values["pressure"]) / len(values["pressure"]),
            "humidity_mean": sum(values["humidity"]) / len(values["humidity"]),
            "has_晴": int("Clear" in values["weathers"]),
            "has_雲": int("Clouds" in values["weathers"]),
            "has_雷": int("Thunderstorm" in values["weathers"]),
        }
        records.append(record)

    return pd.DataFrame(records)

# 天気情報以外の特徴量を追加
def add_features(df):
    df["is_holiday"] = df["date"].apply(lambda x: x in japan_holidays.HOLIDAYS)
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

@app.route(route="BeerPredictAPI")
def BeerPredictAPI(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Predicting beer demand for all beer types.')

    try:
        # 使うモデルの種類（ファイル名の接頭辞）
        beer_types = ["PALE", "IPA", "WEIZEN", "STOUT", "PILSNER", "LAGER"]
        model_dir = "models"  # モデル保存フォルダ

        # 天気取得・特徴量作成（共通処理）
        df = get_weather_forecast()
        df = add_features(df)

        all_predictions = {}

        for beer in beer_types:
            model_path = f"{model_dir}/{beer}_best_model.pkl"
            try:
                data = joblib.load(model_path)
                model = data["model"]
                scaler = data["scaler"]

                # 前処理と推論
                X_processed = preprocess(df.copy(), scaler)
                predictions = model.predict(X_processed)

                # 結果格納（ビールごとに日付→予測量）
                all_predictions[beer] = {
                    str(date): round(pred, 2)
                    for date, pred in zip(df["date"], predictions)
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
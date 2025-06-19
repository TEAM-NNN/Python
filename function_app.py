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
    df["is_holiday"] = df["date"].apply(jpholiday.is_holiday)
    df['is_holiday'] = df['is_holiday'].astype(int)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["weekday"] = df["date"].apply(lambda x: x.weekday())
    return df

# APIエンドポイント
@app.function_name(name="predapi") 
@app.route(route="predapi")
def predapi(req: func.HttpRequest) -> func.HttpResponse:
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

        # 天気情報取得
        df = get_weather_forecast()

        # 曜日で対象日を抽出
        today = datetime.now().date()
        weekday = today.weekday()  # 月=0, 木=3 など
        if weekday == 0:  # 月曜
            target_dates = [today + timedelta(days=i) for i in [1, 2, 3]]
        elif weekday == 3:  # 木曜
            target_dates = [today + timedelta(days=i) for i in [1, 2, 4]]
        else:
            target_dates = [today + timedelta(days=i) for i in [1, 2, 3]]

        df = df[df["date"].isin(target_dates)].reset_index(drop=True)
        df = add_features(df)

        # One-hot エンコード
        df_encoded = pd.get_dummies(df, columns=["month", "weekday"], prefix=["month", "weekday"], drop_first=False)
        df_encoded.drop(columns=["month_1", "weekday_0"], errors="ignore", inplace=True)

        expected_dummies = [
            "month_2", "month_3", "month_4", "month_5", "month_6",
            "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
            "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5"
        ]
        for col in expected_dummies:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
        df_encoded[expected_dummies] = df_encoded[expected_dummies].astype(float)

        # モデルを一括ロード
        models = {}
        for beer in beer_types:
            model_path = f"{model_dir}/{beer}_best_model.pkl"
            try:
                data = joblib.load(model_path)
                models[beer] = {
                    "model": data["model"],
                    "scaler": data["scaler"]
                }
            except Exception as e:
                logging.warning(f"Could not load model for {beer}: {e}")
                models[beer] = None

        result = []
        for beer in beer_types:
            if models[beer] is None:
                result.append({"beerName": beer.replace("_", ""), "quantity": 0})
                continue

            features = feature_map[beer]
            missing = [col for col in features if col not in df_encoded.columns]
            if missing:
                return func.HttpResponse(f"{beer} の欠損特徴量: {missing}", status_code=200)

            X = df_encoded[features]
            X_scaled = models[beer]["scaler"].transform(X)
            preds = models[beer]["model"].predict(X_scaled)
            total = int(np.sum(np.ceil(preds)))

            result.append({
                "beerName": {
                    "PALE": "ペールエール",
                    "LAGER": "ラガービール",
                    "IPA": "IPA",
                    "WHITE": "ホワイトビール",
                    "BLACK": "黒ビール",
                    "FRUIT": "フルーツビール"
                }[beer],
                "quantity": total
            })

        return func.HttpResponse(
            body=json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Fatal error occurred: {e}")
        return func.HttpResponse(f"Internal error: {e}", status_code=500)

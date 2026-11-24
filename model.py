import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

class TrafficETAModel:
    def __init__(self):
        self.model = None
        self.hist_lookup = None
        self.global_median_speed = 18.0  # km/h fallback for Delhi buses

    def prepare_features(self, df, hist_stats):
        df = df.merge(hist_stats, on=["segment_id", "hour", "day_of_week"], how="left")
        df["hist_travel_time"] = df["hist_travel_time"].fillna(
            (df["distance_m"] / 1000.0) / self.global_median_speed * 3600.0
        )
        # If live speed isn't available for a segment, fall back to historical speed
        hist_speed = (df["distance_m"] / 1000.0) / (df["hist_travel_time"] / 3600.0)
        df["recent_live_speed"] = df["recent_live_speed"].fillna(hist_speed)
        df["speed_ratio"] = df["recent_live_speed"] / np.maximum(hist_speed, 1.0)
        df["segment_code"] = df["segment_id"].astype("category")
        return df

    def train(self, obs_df):
        # 1. Build Historical Baseline (Hour x Day of Week)
        self.hist_lookup = (
            obs_df.groupby(["segment_id", "hour", "day_of_week"])["travel_time_sec"]
            .median()
            .reset_index()
            .rename(columns={"travel_time_sec": "hist_travel_time"})
        )

        # Simulate recent_live_speed during training using rolling vehicle speeds
        obs_df = obs_df.sort_values("timestamp").copy()
        obs_df["recent_live_speed"] = (
            obs_df.groupby("segment_id")["speed_kmph"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

        feat_df = self.prepare_features(obs_df, self.hist_lookup)
        features = ["segment_code", "hour", "day_of_week", "distance_m", "hist_travel_time", "recent_live_speed", "speed_ratio"]

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        self.model.fit(feat_df[features], feat_df["travel_time_sec"])

    def predict_segments(self, segments_df, hour, day_of_week, live_speeds_map=None):
        """Returns predicted travel_time_sec and congestion level for all segments."""
        df = segments_df.copy()
        df["hour"] = hour
        df["day_of_week"] = day_of_week
        df["recent_live_speed"] = df["segment_id"].map(live_speeds_map or {})

        feat_df = self.prepare_features(df, self.hist_lookup)
        features = ["segment_code", "hour", "day_of_week", "distance_m", "hist_travel_time", "recent_live_speed", "speed_ratio"]

        preds = self.model.predict(feat_df[features])
        df["pred_travel_time_sec"] = np.maximum(preds, 15.0)  # minimum 15s per segment
        df["pred_speed_kmph"] = (df["distance_m"] / 1000.0) / (df["pred_travel_time_sec"] / 3600.0)

        # Congestion classification for frontend color overlay
        df["congestion"] = np.where(
            df["pred_speed_kmph"] < 12.0, "heavy",
            np.where(df["pred_speed_kmph"] < 20.0, "moderate", "free_flow")
        )
        return df

    def save(self, path="artifacts/eta_model.joblib"):
        joblib.dump({"model": self.model, "hist_lookup": self.hist_lookup}, path)

    def load(self, path="artifacts/eta_model.joblib"):
        data = joblib.load(path)
        self.model = data["model"]
        self.hist_lookup = data["hist_lookup"]

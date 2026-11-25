import json
import logging
import os
import shutil
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(cfg):
    """Загружает признаки и целевую переменную, соединяет по subscriber_key."""
    features = pd.read_parquet(cfg["data"]["features_path"])
    churn = pd.read_csv(cfg["data"]["churn_path"])
    churn["churned"] = churn["churned"].astype(int)
    df = features.merge(churn[["subscriber_key", "churned"]], on="subscriber_key")
    logging.info("Загружено абонентов: %d", len(df))
    return df


def preprocess(df, cfg):
    """Делит на train/val/test, заполняет пропуски, масштабирует."""
    p = cfg["preprocessing"]
    cols = [c for c in df.select_dtypes("number").columns if c not in p["exclude_columns"]]
    X, y = df[cols].fillna(0), df["churned"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=p["test_size"], random_state=p["random_state"], stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=p["val_size"], random_state=p["random_state"], stratify=y_tmp)
    scaler = StandardScaler().fit(X_tr)
    logging.info("Признаков: %d, train/val/test: %d/%d/%d", len(cols), len(y_tr), len(y_val), len(y_te))
    return scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te), y_tr, y_val, y_te, cols, scaler


def train_model(X_train, y_train, cfg):
    """Обучает RandomForest с параметрами из конфига."""
    model = RandomForestClassifier(**cfg["model"]).fit(X_train, y_train)
    logging.info("Модель обучена")
    return model


def evaluate(model, X_val, y_val, X_test, y_test):
    """Считает ROC-AUC на val и test, F1 на test."""
    metrics = {
        "val_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
        "test_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "test_f1": f1_score(y_test, model.predict(X_test)),
    }
    logging.info("Метрики: %s", metrics)
    return metrics


def save_artifacts(model, scaler, cols, metrics, cfg):
    """Сохраняет артефакты. Идемпотентно: старая папка удаляется перед записью."""
    a = cfg["artifacts"]
    if os.path.exists(a["model_dir"]):
        shutil.rmtree(a["model_dir"])
    os.makedirs(a["model_dir"])
    joblib.dump(model, os.path.join(a["model_dir"], a["model_name"]))
    joblib.dump(scaler, os.path.join(a["model_dir"], a["scaler_name"]))
    with open(os.path.join(a["model_dir"], a["features_name"]), "w") as f:
        json.dump(cols, f, indent=2)
    with open(os.path.join(a["model_dir"], a["metrics_name"]), "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Артефакты сохранены в %s", a["model_dir"])


def main(config_path="config.json"):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    logging.basicConfig(level=cfg["logging"]["level"], format=cfg["logging"]["format"],
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(cfg["logging"]["log_file"], mode="w", encoding="utf-8")])
    start = time.time()
    try:
        df = load_data(cfg)
        X_tr, X_val, X_te, y_tr, y_val, y_te, cols, scaler = preprocess(df, cfg)
        model = train_model(X_tr, y_tr, cfg)
        metrics = evaluate(model, X_val, y_val, X_te, y_te)
        save_artifacts(model, scaler, cols, metrics, cfg)
    except Exception:
        logging.exception("Ошибка в конвейере")
        raise
    logging.info("Готово за %.1f с", time.time() - start)


if __name__ == "__main__":
    main()

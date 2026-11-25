# Прогноз оттока абонентов: конвейер

## Описание задачи
По признакам абонента (звонки, обращения в поддержку, телеметрия узла связи) предсказать отток (`churned`). Данные синтетические (уровень XS). Модель — RandomForest. Этапы: `load_data → preprocess → train_model → evaluate → save_artifacts` (схема в `architecture.png`). Все параметры — в `config.json`.

## Установка зависимостей
Нужен Python 3.13.
```bash
pip install -r requirements.txt
```

## Запуск конвейера
```bash
python main.py
```
Запуск в Docker:
```bash
docker build -t churn-pipeline .
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" churn-pipeline
```

## Артефакты
В папке `artifacts/` (при каждом запуске пересоздаётся заново):
- `best_model.pkl` — обученная модель;
- `scaler.pkl` — StandardScaler;
- `features.json` — список признаков;
- `metrics.json` — метрики (val AUC, test AUC, test F1).

Журнал запуска — `pipeline.log`.

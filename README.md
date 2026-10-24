# Прогноз отказов оборудования — Яньшин А.

**Автор доработки: Яньшин А.** Дисциплина: «Бизнес-применение машинного обучения».
Основа — форк группового проекта и код `main` на коммите `41b06a6`.
История разработки сохранена. Все изменения для этой сдачи находятся в
[ветке yanshin](https://github.com/svscrip/auto_ML_prediction_fail_new/tree/yanshin).

![CI yanshin](https://github.com/svscrip/auto_ML_prediction_fail_new/actions/workflows/ci.yml/badge.svg?branch=yanshin)

## Материалы для сдачи

| Задание | Результат |
| --- | --- |
| 1. Рефакторинг, Poetry, pre-commit, окружение | [Изменения и результаты проверок](docs/assignments.md) |
| 2. Описание модели | [model_card.md](model_card.md) |
| 2. Роль в бизнес-процессе | [Model_BPMN.md](Model_BPMN.md) |
| 3. Процесс переобучения | [Model_retrain_BPMN.md](Model_retrain_BPMN.md) |
| Индивидуальная защита | [Речь, демонстрация и ответы на вопросы](docs/defense.md) |

## Независимая перепроверка

[Задание другой нейросети и критерии аудита](docs/AI_REVIEW.md).

```powershell
poetry run python scripts/review_project.py verify
poetry run python scripts/review_project.py bundle --verification artifacts/review/verification-ИМЯ_ПРОГОНА
```

Первая команда создаёт логи и JSON-протокол в отдельном хранилище, вторая —
TXT и ZIP для передачи другому рецензенту. Путь прогона выводится после проверки.
Автоматической отправки внешним сервисам нет.

## Что делает проект

Пакетный Python-пайплайн загружает `train.csv` / `test.csv`, проверяет данные,
строит признаки, обучает CatBoost, сохраняет эксперименты в MLflow и формирует
оценки отказа с рекомендациями. Данные — синтетический
[Kaggle Playground S3E17](https://www.kaggle.com/competitions/playground-series-s3e17/data).

Модель — **учебный baseline**: среди признаков есть флаги уже произошедшего
отказа. Раннее предупреждение на реальном оборудовании пока не доказано.
Нужны признаки без утечки, временной протокол и независимая оценка.
Ограничения и исторические результаты подробно раскрыты в карточке модели.

## Установка и воспроизводимое окружение

Требования: Git, Python 3.11–3.13 и Poetry 2.x. Проверено локально на Python 3.12.

```powershell
git clone --branch yanshin https://github.com/svscrip/auto_ML_prediction_fail_new.git
cd auto_ML_prediction_fail_new
poetry env use python
poetry sync --with dev
poetry run pre-commit install
poetry check --lock
```

`pyproject.toml` описывает зависимости, `poetry.lock` фиксирует версии,
`poetry.toml` задаёт `.venv` внутри проекта. Каталог `.venv` исключён из Git:
на другой машине окружение восстанавливается из lock-файла.
Активация необязательна, команды можно выполнять через `poetry run`.

## Проверки

```powershell
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run pytest -q --override-ini addopts=''
poetry run python -m src.model_card --check model_card.md --metrics artifacts/example_model_card_metrics.json
```

Последняя pytest-команда включает slow smoke-тест обучения. Тесты обучения
пишут модель и MLflow-запуски во временный каталог pytest, поэтому не заменяют
`artifacts/model.cbm` и не добавляют запуски в рабочую историю. Black, isort и
Flake8 проверяют стиль; pytest проверяет ETL, обучение, мониторинг и карточку.
CI запускается при push в `yanshin`, включает Python 3.11/3.12/3.13 и Docker smoke.
Публикация релизов по тегам — отдельный исходный workflow; тег для сдачи не нужен.

## Обучение и инференс

```powershell
poetry run python -m src.train --smoke
poetry run python -m src.predict
```

Полное обучение: `poetry run python -m src.train`. Smoke использует 5 000 строк;
его метрики нельзя выдавать за полный прогон. По умолчанию результат обучения
перезаписывает `artifacts/model.cbm`, а MLflow сохраняет отдельный run.
Не используйте этот локальный путь как хранилище действующей production-модели.

После запуска в `artifacts/` появляются:

- `model.cbm`, `metrics.json`, `plots/` — модель, метрики и графики;
- `data_quality.json`, `monitoring_summary.json` — отчёты обучения;
- `predictions.csv`, `maintenance_recommendations.csv`, `inference_monitoring.json` — инференс;
- `mlflow.db`, `mlartifacts/` — локальная история экспериментов.

`risk_level` основан на квантилях конкретного пакета. Это не абсолютная
вероятность и не порог 0.5, используемый для бинарных метрик.

Запуск интерфейса MLflow в Windows:

```powershell
.\scripts\start-mlflow.ps1
```

После старта открыть `http://127.0.0.1:5000`; остановка — Ctrl+C.

## Конфигурация

См. [.env.example](.env.example). Значения задаются в окружении процесса;
файл `.env` автоматически не загружается. Относительные пути отсчитываются
от текущего каталога, команды выше запускаются из корня репозитория.

```powershell
$env:ML_DATA_DIR = '.\keis7-main'
$env:ML_ARTIFACTS_DIR = '.\artifacts\experiment-yanshin'
$env:MLFLOW_EXPERIMENT_NAME = 'machine_failure_prediction_yanshin'
poetry run python -m src.train --smoke
poetry run python -m src.predict
```

Также поддержаны `MLFLOW_TRACKING_URI` и `MLFLOW_ARTIFACTS_URI`. Для удалённого
tracking путь артефактов должен быть доступен серверу и клиентам по согласованной
схеме хранения. При конфликте сохранённого пути эксперимент не удаляется:
исправьте конфигурацию или выберите отдельное имя/хранилище, сохранив историю.
Секреты не помещаются в код или Git.

## Docker

Требуется работающий Docker Engine; для локальных Python-проверок он не нужен.

```powershell
docker build -t machine-failure-mlops:yanshin .
docker run --rm -v yanshin-artifacts:/app/artifacts machine-failure-mlops:yanshin src.train --smoke
docker run --rm -v yanshin-artifacts:/app/artifacts machine-failure-mlops:yanshin src.predict
```

Образ использует непривилегированного пользователя; артефакты сохраняются в томе.

`docker compose up train predict mlflow` монтирует `./artifacts`, но ведёт
отдельное хранилище в `artifacts/docker/` (модель, `mlflow.db`, артефакты).
Локальная база `artifacts/mlflow.db` хранит пути хоста, поэтому контейнер её
не использует: иначе `src.mlflow_setup` остановил бы обучение из-за конфликта
путей. Интерфейс MLflow из compose показывает только контейнерные запуски.
Состояние проверки контейнера отражается отдельной задачей CI.

## Карточка и процессы

Числовые таблицы карточки воспроизводятся из сохранённого JSON-снимка:

```powershell
poetry run python -m src.model_card --write model_card.md --metrics artifacts/example_model_card_metrics.json
```

Это исторический снимок полного прогона исходной версии, с его SHA и окружением.
Чтобы получить новый снимок, сначала выполнить полное обучение, затем
`poetry run python -m src.model_card --output artifacts/model_card_metrics.json`.
Перед заменой публикуемых чисел нужно обновить метаданные и сравнить результаты.

Схема переобучения существует в Markdown и [BPMN 2.0](docs/bpmn/retraining-yanshin.bpmn)
для bpmn.io. Это проектируемый процесс: автоматический scheduler, Registry,
согласование, Telegram и промышленный rollout пока не подключены.

## Структура

- `src/` — загрузка, признаки, CatBoost, инференс, мониторинг, генератор карточки.
- `tests/` — проверки вычислений, ошибочных входов и документации.
- `keis7-main/` — исходные CSV.
- `docs/` — материалы заданий, речи, BPMN и метаданные карточки.
- `reference-material/` — архив исходного группового исследования.
- `scripts/` — вспомогательные команды Windows.
- `.github/workflows/` — CI и исходный workflow релизов.

Правила разработки: [CONTRIBUTING.md](CONTRIBUTING.md).
Происхождение данных и применимость лицензий проверяются по первоисточникам;
лицензия UCI не распространяется автоматически на весь код и конкурсный набор.

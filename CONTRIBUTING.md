# Contributing

Спасибо за интерес к проекту! Ниже — правила разработки и настройка окружения.

## 1. Требования

- **Python 3.11+** (проект поддерживает 3.11, 3.12, 3.13).
- **Poetry 2.x** — управление зависимостями и виртуальным окружением.
- **Git**.
- _(опционально)_ **Docker** — для запуска пайплайна в контейнере.

## 2. Быстрая настройка окружения

Проект использует **per-project виртуальное окружение** (`.venv/` в корне
репозитория), поэтому Poetry не трогает глобальные окружения.

### Windows (PowerShell)

```powershell
# Установить Poetry (если ещё не установлен)
pip install --user poetry

# Создать .venv и установить зависимости (runtime + dev)
poetry install

# Активировать окружение
.\.venv\Scripts\Activate.ps1
```

Либо одной командой через скрипт-хелпер:

```powershell
.\scripts\bootstrap.ps1     # установка Poetry + poetry install
.\scripts\activate.ps1      # активация .venv
```

### Linux / macOS

```bash
pip install --user poetry
poetry install
source .venv/bin/activate
```

> Преференции окружения фиксируются в `poetry.toml` (`virtualenvs.in-project = true`),
> поэтому `.venv` всегда создаётся внутри репозитория и игнорируется git.

## 3. Установка git-хуков (pre-commit)

```bash
poetry run pre-commit install
```

Теперь при каждом `git commit` автоматически запускаются:

- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`,
  `check-added-large-files`, `detect-private-key`, `mixed-line-ending`;
- **black** — форматирование кода;
- **isort** — сортировка импортов;
- **flake8** (+ flake8-bugbear) — статический анализ.

Один прогон по всем файлам:

```bash
poetry run pre-commit run --all-files
```

## 4. Стиль кода

| Инструмент | Назначение | Конфигурация |
|------------|------------|--------------|
| **Black** | автоформат, длина строки 88 | `pyproject.toml → [tool.black]` |
| **isort** | сортировка импортов (profile=black) | `pyproject.toml → [tool.isort]` |
| **flake8** | линтер (PEP8 + bugbear) | `.flake8` |

Правила:

- длина строки — **88 символов**;
- одинарные/двойные кавычки нормализуются Black;
- приватные хелперы — с префиксом `_`;
- все публичные функции сопровождаются docstring.

Проверить локально без коммита:

```bash
poetry run black --check .
poetry run isort --check-only .
poetry run flake8
```

## 5. Тесты

```bash
poetry run pytest -q                 # быстрые тесты (без маркера slow)
poetry run pytest -q -m slow         # полный smoke-тест обучения
poetry run pytest -q --cov=src       # с покрытием
```

Пометкой `@pytest.mark.slow` отмечены долгие тесты (полное обучение);
по умолчанию они исключаются через `addopts`.

## 6. Git workflow

- Базовая ветка — `main`.
- Новая функциональность — в ветке `feature/<краткое-имя>`.
- Изменения вливаются через **Pull Request**; merge — только после зелёного CI.
- Сообщения коммитов — в формате
  [Conventional Commits](https://www.conventionalcommits.org/):
  `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`, `refactor: ...`, `test: ...`.

## 7. CI/CD

- **CI** (`.github/workflows/ci.yml`): на каждый push/PR — линтинг (pre-commit),
  тесты в матрице Python 3.11/3.12/3.13 и Docker smoke-прогон.
- **CD** (`.github/workflows/cd.yml`): по тегу `v*` — сборка и публикация
  Docker-образа в GHCR + GitHub Release.

## 8. Зависимости

- **Источник истины — `poetry.lock`** (коммитится вместе с проектом).
- Добавление зависимости:
  `poetry add <pkg>` (runtime) / `poetry add --group dev <pkg>` (dev).
- Обновление `requirements.txt` (pinned runtime) для Docker:

  ```bash
  poetry export --without-hashes --only main -o requirements.txt
  poetry export --without-hashes --with dev -o requirements-lock.txt
  ```

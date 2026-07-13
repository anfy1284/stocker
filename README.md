# Stocker

Автоматизированный отбор и загрузка фотографий на микростоки. Модульный
конвейер поверх общей SQLite-БД: приём → классификация «сток/не-сток» →
группировка серий и отбор лучших → веб-ревью → метаданные → загрузка.

Полное описание — в [`ТЗ.md`](ТЗ.md); план по шагам — в
[`ПЛАН_РЕАЛИЗАЦИИ.md`](ПЛАН_РЕАЛИЗАЦИИ.md); техника — в
[`АРХИТЕКТУРА_ПРОЕКТА.md`](АРХИТЕКТУРА_ПРОЕКТА.md).

## Установка (Windows 11)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # затем впишите ANTHROPIC_API_KEY
```

## Запуск

```powershell
python -m stocker          # init: создаёт каталоги data/ и пустую БД
python -m stocker config   # показать текущую конфигурацию
```

После `init` появляются `data/inbox`, `data/previews`, `data/logs` и файл
`data/stocker.db`. Пути и ключ настраиваются через `.env` (см. `.env.example`).

## Статус

Реализован **Шаг 1** — каркас и конфигурация. Дальше: приём JPEG (Шаг 2).

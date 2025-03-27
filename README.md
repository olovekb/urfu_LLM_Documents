# Работа для архива
## Требуется реализовать ТЗ (см. КП)

Запуск web-api на streamlit:
'''
python -m streamlit run ./src/app/streamlit/main.py
'''
Запуск сервисов
1. Файл конфигурации /src/app/config.yaml
    fastapi - локальный адрес:порт для api сервиса проверки 
    root_path - внеший адрес в сети при использовании через nginx, по умолчанию ""
2. Сборка докера docker compose build

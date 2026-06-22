@echo off
echo Запуск MedVision...
cd /d "%~dp0"

echo Открываем Edge на http://localhost:8000 ...
start msedge http://localhost:8000

echo Запускаем Docker Compose...
docker compose up --build

pause
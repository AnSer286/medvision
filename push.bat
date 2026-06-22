@echo off
echo ================================
echo PUSH изменений в репозиторий
echo ================================
cd /d "%~dp0"

REM Получаем текущую ветку
for /f %%i in ('git branch --show-current') do set current_branch=%%i
echo Текущая ветка: %current_branch%

REM Показываем статус
echo.
echo Статус изменений:
git status --short

echo.
set /p commit_msg="Введите сообщение коммита (или нажмите Enter для 'Обновление'): "
if "%commit_msg%"=="" set commit_msg=Обновление

echo.
echo Добавляем все изменения...
git add .

echo Коммитим с сообщением: "%commit_msg%"
git commit -m "%commit_msg%"

echo Пушим в origin/%current_branch%...
git push origin %current_branch%

echo.
echo Готово!
pause
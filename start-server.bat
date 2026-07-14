@echo off
rem Запуск веб-интерфейса ревью Stocker.
rem Работает из любого места — переходит в папку самого батника.
cd /d "%~dp0"
echo Запуск сервера... Открой в браузере http://localhost:8000
echo (с телефона в той же сети — http://<IP-этого-ПК>:8000)
echo Останов — закрой это окно или нажми Ctrl+C.
echo.
".venv\Scripts\python.exe" -m stocker web
echo.
echo Сервер остановлен.
pause

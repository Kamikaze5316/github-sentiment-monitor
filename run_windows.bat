@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       GitHub Sentiment Monitor
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto setup_error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto setup_error

echo.
echo Installing or checking dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto setup_error
python -m pip install -r requirements.txt
if errorlevel 1 goto setup_error

echo.
echo Starting Streamlit. Keep this window open while using the app.
echo.
python -m streamlit run app.py
if errorlevel 1 goto run_error
goto end

:setup_error
echo.
echo Setup failed. Check that Python 3.11 is installed and available in PATH.
pause
exit /b 1

:run_error
echo.
echo Streamlit stopped with an error. Review the message above.
pause
exit /b 1

:end
pause

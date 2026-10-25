@echo off
title 5G Traffic Prediction Web App
echo ==========================================
echo     5G Traffic Prediction Web App
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

echo Activating environment...
call .venv\Scripts\activate.bat

echo Installing/checking dependencies...
python -m pip install -r requirements.txt

echo.
echo Starting Streamlit...
python -m streamlit run app.py

pause

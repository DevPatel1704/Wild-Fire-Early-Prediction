@echo off
echo ============================================
echo  Wildfire IoT Early Warning System - Setup
echo ============================================

echo.
echo [1/5] Copying .env.example to .env ...
if not exist .env (
    copy .env.example .env
    echo .env created. Edit it if needed.
) else (
    echo .env already exists.
)

echo.
echo [2/5] Installing Python dependencies ...
pip install -r requirements.txt

echo.
echo [3/5] Starting Docker services (Kafka, InfluxDB) ...
docker-compose up -d
echo Waiting 20 seconds for services to start...
timeout /t 20 /nobreak >nul

echo.
echo [4/5] Installing React dashboard dependencies ...
cd dashboard
call npm install
cd ..

echo.
echo [5/5] Setup complete!
echo.
echo Next steps:
echo   1. Generate training data:  python -m simulator.run_simulator --fast --export csv --days 30
echo   2. Preprocess data:         python -m data.preprocess
echo   3. Train model:             python -m model.train --epochs 50
echo   4. Start API server:        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
echo   5. Start dashboard:         cd dashboard ^&^& npm start
echo   6. Start live simulator:    python -m simulator.run_simulator
echo   7. Start stream processor:  python -m pipeline.stream_processor
echo.
echo Dashboard will be available at: http://localhost:3000
echo API docs will be at:            http://localhost:8000/docs
echo Kafka UI will be at:            http://localhost:8080
echo InfluxDB UI will be at:         http://localhost:8086

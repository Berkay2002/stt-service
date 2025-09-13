# test_endpoints.bat

@echo off
echo Testing STT Service Monitoring Endpoints
echo ========================================

echo.
echo Starting monitoring server in background...
start /B python -m app.main --monitor

echo Waiting 5 seconds for server to start...
timeout /T 5 /NOBREAK > NUL

echo.
echo Testing endpoints with curl...
echo.

echo 📊 Health Check:
curl -s http://localhost:9091/health | python -m json.tool
echo.

echo 📈 Metrics:
curl -s http://localhost:9091/metrics | python -m json.tool
echo.

echo ℹ️ Service Info:
curl -s http://localhost:9091/info | python -m json.tool
echo.

echo 🔍 Readiness Check:
curl -s http://localhost:9091/health/ready | python -m json.tool
echo.

echo ❤️ Liveness Check:
curl -s http://localhost:9091/health/live | python -m json.tool

echo.
echo ========================================
echo Tests completed!
echo Note: Monitoring server is still running in background
echo Use Ctrl+C to stop it, or check Task Manager
@echo off
chcp 65001 >nul
echo ========================================
echo Start CARLA Server - High Performance
echo ========================================
echo.
cd /d D:\carla\CARLA_0.9.16
if not exist CarlaUE4.exe (
  echo [ERROR] CarlaUE4.exe not found in D:\carla\CARLA_0.9.16
  pause
  exit /b 1
)
echo Launching CarlaUE4 (Epic + Vulkan)...
CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Epic -vulkan -fps=120 -NoVSync -Windowed -ResX=1280 -ResY=720
pause

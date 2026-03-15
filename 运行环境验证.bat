@echo off
chcp 65001 >nul
echo ========================================
echo CARLA环境验证
echo ========================================
echo.
cd /d C:\CARLA
python 环境验证脚本.py
pause

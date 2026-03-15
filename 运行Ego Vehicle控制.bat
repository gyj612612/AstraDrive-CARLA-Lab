@echo off
chcp 65001 >nul
echo ========================================
echo CARLA Ego Vehicle控制
echo ========================================
echo.
echo 此脚本将创建Ego Vehicle并允许您手动控制
echo.
echo 控制说明：
echo   W/↑: 加速
echo   S/↓: 刹车
echo   A/←: 左转
echo   D/→: 右转
echo   ESC: 退出
echo.
echo 注意：请先启动CARLA服务器（运行"启动CARLA服务器.bat"）
echo.
pause
cd /d C:\CARLA
python control_ego_vehicle.py
pause


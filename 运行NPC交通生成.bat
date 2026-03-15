@echo off
chcp 65001 >nul
echo ========================================
echo CARLA NPC交通生成
echo ========================================
echo.
echo 此脚本将生成NPC车辆和行人，让它们自动运行
echo.
echo 注意：请先启动CARLA服务器（运行"启动CARLA服务器.bat"）
echo.
pause
cd /d C:\CARLA
python spawn_npc_traffic.py -n 30 -w 20
pause


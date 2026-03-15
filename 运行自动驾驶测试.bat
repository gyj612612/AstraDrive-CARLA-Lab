@echo off
chcp 65001 >nul
echo ========================================
echo CARLA自动驾驶和多模态数据采集测试
echo ========================================
echo.
echo 此脚本将：
echo 1. 生成NPC车辆和行人（自动运行）
echo 2. 创建Ego Vehicle（我们控制的车辆）
echo 3. 在Ego Vehicle上安装多模态传感器
echo 4. 采集和保存多模态数据
echo.
echo 注意：请先启动CARLA服务器（运行"启动CARLA服务器.bat"）
echo.
pause
cd /d C:\CARLA
python autonomous_driving_test.py --num-vehicles 30 --num-walkers 20
pause

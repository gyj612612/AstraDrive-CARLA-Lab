@echo off
chcp 65001 >nul
echo ========================================
echo 运行CARLA手动控制示例
echo ========================================
echo.
echo 注意：请先启动CARLA服务器（运行"启动CARLA服务器.bat"）
echo.
echo 控制说明：
echo - 方向键：控制车辆
echo - 鼠标：控制视角
echo - ESC：退出
echo.
cd /d D:\carla\CARLA_0.9.16\PythonAPI\examples
python manual_control.py
pause


@echo off
chcp 65001 >nul
echo ========================================
echo 运行CARLA交通生成示例
echo ========================================
echo.
echo 注意：请先启动CARLA服务器（运行"启动CARLA服务器.bat"）
echo.
cd /d D:\carla\CARLA_0.9.16\PythonAPI\examples
python generate_traffic.py
pause


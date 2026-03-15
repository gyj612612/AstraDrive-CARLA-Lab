@echo off
chcp 65001 >nul
echo ========================================
echo 启动CARLA服务器
echo ========================================
echo.
cd /d D:\carla\CARLA_0.9.16
echo 正在启动CARLA服务器...
echo 服务器启动后，将显示一个3D城市视图窗口
echo 您可以使用鼠标和WASD键在观察者视图中飞行
echo.
echo 按Ctrl+C可以停止服务器
echo.
CarlaUE4.exe
pause


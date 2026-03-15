# CARLA 0.9.16 安装配置说明

## 安装完成情况

✅ **CARLA Python客户端库** - 已安装（版本 0.9.16）  
✅ **示例脚本依赖包** - 已安装  
✅ **安装验证** - 通过

## 安装位置

- **CARLA主程序**: `D:\carla\CARLA_0.9.16\`
- **CARLA服务器**: `D:\carla\CARLA_0.9.16\CarlaUE4.exe`
- **Python API**: `D:\carla\CARLA_0.9.16\PythonAPI\`
- **示例脚本**: `D:\carla\CARLA_0.9.16\PythonAPI\examples\`

## 已安装的Python包

- carla (0.9.16)
- future
- numpy
- networkx
- Shapely
- pygame
- matplotlib
- open3d
- Pillow

**注意**: `invertedai` 包由于依赖问题未安装，但这不是必需的，不影响基本使用。

## 快速开始

### 1. 启动CARLA服务器

双击运行 `启动CARLA服务器.bat`，或者手动执行：

```bash
cd D:\carla\CARLA_0.9.16
CarlaUE4.exe
```

服务器启动后会显示一个3D城市视图窗口。这是CARLA的观察者视图，您可以使用：
- **鼠标右键拖拽**: 旋转视角
- **WASD键**: 移动视角
- **鼠标滚轮**: 缩放

### 2. 运行示例脚本

在服务器运行后，打开新的命令行窗口运行示例：

#### 生成交通（终端A）
双击运行 `运行交通生成示例.bat`，或执行：
```bash
cd D:\carla\CARLA_0.9.16\PythonAPI\examples
python generate_traffic.py
```

#### 手动控制（终端B）
双击运行 `运行手动控制示例.bat`，或执行：
```bash
cd D:\carla\CARLA_0.9.16\PythonAPI\examples
python manual_control.py
```

手动控制窗口打开后，您可以使用方向键控制车辆在CARLA世界中行驶。

## 基本Python代码示例

创建一个新的Python脚本连接到CARLA：

```python
import carla

# 连接到CARLA服务器（默认localhost:2000）
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取世界对象
world = client.get_world()

# 获取当前地图
map = world.get_map()

# 获取所有生成点
spawn_points = map.get_spawn_points()
print(f"地图中有 {len(spawn_points)} 个生成点")

# 获取所有蓝图
blueprint_library = world.get_blueprint_library()

# 查找车辆蓝图
vehicle_bp = blueprint_library.filter('vehicle.*')[0]

# 在第一个生成点生成车辆
spawn_point = spawn_points[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print(f"已生成车辆: {vehicle.type_id}")
```

## 端口配置

CARLA默认使用以下端口：
- **2000**: 主要通信端口
- **2001**: 流媒体端口

确保这些端口未被防火墙或其他应用程序占用。

## 附加地图

如果您下载了附加地图包（`AdditionalMaps_0.9.16`），需要将其内容复制到CARLA主目录中。请参考CARLA官方文档了解详细的地图安装步骤。

## 常见问题

### 1. 无法连接到服务器
- 确保CARLA服务器已启动
- 检查端口2000和2001是否被占用
- 检查防火墙设置

### 2. 导入carla模块失败
- 确认已正确安装：`python -m pip list | findstr carla`
- 如果失败，重新安装：`python -m pip install D:\carla\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl`

### 3. 示例脚本运行错误
- 确保所有依赖已安装：`python -m pip install future numpy networkx Shapely pygame matplotlib open3d Pillow`
- 确保CARLA服务器正在运行

## 下一步

- 查看CARLA官方文档：https://carla.readthedocs.io/
- 浏览示例脚本：`D:\carla\CARLA_0.9.16\PythonAPI\examples\`
- 学习CARLA Python API：https://carla.readthedocs.io/en/latest/python_api/

## 系统要求

- ✅ 操作系统: Windows 10/11
- ✅ Python: 3.12.7
- ✅ 磁盘空间: 约20GB（已满足）
- ⚠️ GPU: 建议NVIDIA 2070或更好，至少8GB VRAM（请确认您的GPU配置）

---

安装完成时间: 2024年
安装版本: CARLA 0.9.16


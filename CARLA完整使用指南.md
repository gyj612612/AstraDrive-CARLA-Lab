# CARLA 自动驾驶和多模态数据采集完整使用指南

## 📋 目录

1. [环境概述](#环境概述)
2. [快速开始](#快速开始)
3. [脚本说明](#脚本说明)
4. [多模态数据采集](#多模态数据采集)
5. [NPC交通管理](#npc交通管理)
6. [Ego Vehicle控制](#ego-vehicle控制)
7. [常见问题](#常见问题)

---

## 环境概述

### 已安装的组件

✅ **CARLA 0.9.16** (UE4版本) - 位于 `D:\carla\CARLA_0.9.16\`  
✅ **CARLA UE5 源代码** - 位于 `C:\CARLA\carla-ue5-dev\`  
✅ **Python客户端库** - 已安装  
✅ **所有依赖包** - 已安装

### 系统要求

- ✅ 操作系统: Windows 10/11
- ✅ Python: 3.12.7
- ✅ GPU: 建议NVIDIA 2070或更好，至少8GB VRAM
- ✅ 磁盘空间: 约20GB
- ✅ 端口: 2000和2001（确保未被占用）

---

## 快速开始

### 步骤1: 启动CARLA服务器

**方法1: 使用批处理脚本（推荐）**
```
双击运行: 启动CARLA服务器.bat
```

**方法2: 手动启动**
```bash
cd D:\carla\CARLA_0.9.16
CarlaUE4.exe
```

服务器启动后会显示一个3D城市视图窗口。这是CARLA的观察者视图，您可以使用：
- **鼠标右键拖拽**: 旋转视角
- **WASD键**: 移动视角
- **鼠标滚轮**: 缩放

### 步骤2: 运行测试脚本

在服务器运行后，选择以下任一脚本：

#### 选项A: 完整测试（推荐）
```
双击运行: 运行自动驾驶测试.bat
```
这将：
- 生成30辆NPC车辆和20个NPC行人（自动运行）
- 创建Ego Vehicle（我们控制的车辆）
- 安装多模态传感器（RGB、深度、语义分割、激光雷达、雷达）
- 自动采集和保存数据

#### 选项B: 仅生成NPC交通
```
双击运行: 运行NPC交通生成.bat
```

#### 选项C: 仅控制Ego Vehicle
```
双击运行: 运行Ego Vehicle控制.bat
```

---

## 脚本说明

### 1. `autonomous_driving_test.py` - 完整测试脚本

**功能：**
- 生成NPC车辆和行人
- 创建Ego Vehicle
- 安装多模态传感器
- 采集和保存数据

**使用方法：**
```bash
python autonomous_driving_test.py [选项]
```

**主要参数：**
- `--host`: CARLA服务器地址（默认: 127.0.0.1）
- `--port`: CARLA服务器端口（默认: 2000）
- `--num-vehicles`: NPC车辆数量（默认: 30）
- `--num-walkers`: NPC行人数量（默认: 20）
- `--sync`: 使用同步模式（推荐用于数据采集）
- `--autopilot`: Ego Vehicle使用自动驾驶
- `--map`: 指定地图（如Town01, Town10等）

**示例：**
```bash
# 使用同步模式，50辆NPC车辆，Ego Vehicle自动驾驶
python autonomous_driving_test.py --sync --num-vehicles 50 --autopilot

# 指定地图Town10
python autonomous_driving_test.py --map Town10
```

### 2. `spawn_npc_traffic.py` - NPC交通生成脚本

**功能：**
- 快速生成NPC车辆和行人
- 自动启用Traffic Manager

**使用方法：**
```bash
python spawn_npc_traffic.py -n 30 -w 20
```

**参数：**
- `-n, --num-vehicles`: NPC车辆数量
- `-w, --num-walkers`: NPC行人数量
- `--sync`: 使用同步模式

### 3. `control_ego_vehicle.py` - Ego Vehicle控制脚本

**功能：**
- 创建Ego Vehicle
- 手动键盘控制
- 采集RGB摄像头数据

**控制键：**
- `W/↑`: 加速
- `S/↓`: 刹车
- `A/←`: 左转
- `D/→`: 右转
- `ESC`: 退出

---

## 多模态数据采集

### 传感器类型

脚本会在Ego Vehicle上安装以下传感器：

1. **RGB摄像头（前视）**
   - 分辨率: 1920x1080
   - FOV: 110度
   - 位置: 车辆前方2.5米，高度0.7米
   - 数据格式: PNG图像
   - 保存位置: `carla_data/rgb/`

2. **深度摄像头**
   - 分辨率: 1920x1080
   - FOV: 110度
   - 数据格式: PNG图像（对数深度）
   - 保存位置: `carla_data/depth/`

3. **语义分割摄像头**
   - 分辨率: 1920x1080
   - FOV: 110度
   - 数据格式: PNG图像（CityScapes调色板）
   - 保存位置: `carla_data/semantic/`

4. **激光雷达**
   - 通道数: 64
   - 点云频率: 1000000点/秒
   - 旋转频率: 20Hz
   - 范围: 100米
   - 位置: 车辆顶部
   - 数据格式: PLY点云文件
   - 保存位置: `carla_data/lidar/`

5. **雷达**
   - 水平FOV: 30度
   - 垂直FOV: 30度
   - 点云频率: 1500点/秒
   - 位置: 车辆前方2.5米
   - 数据格式: TXT文本文件
   - 保存位置: `carla_data/radar/`

### 数据保存结构

```
carla_data/
├── rgb/              # RGB图像
│   ├── rgb_front_000000.png
│   ├── rgb_front_000001.png
│   └── ...
├── depth/            # 深度图像
│   ├── depth_front_000000.png
│   └── ...
├── semantic/         # 语义分割图像
│   ├── semantic_front_000000.png
│   └── ...
├── lidar/            # 激光雷达点云
│   ├── lidar_top_000000.ply
│   └── ...
└── radar/            # 雷达数据
    ├── radar_front_000000.txt
    └── ...
```

### 数据采集模式

#### 异步模式（默认）
- CARLA服务器自动运行
- 传感器数据异步采集
- 适合实时交互

#### 同步模式（推荐用于数据采集）
- 使用 `--sync` 参数启用
- 精确控制每一帧
- 确保传感器数据同步
- 适合机器学习训练数据采集

**使用同步模式：**
```bash
python autonomous_driving_test.py --sync
```

---

## NPC交通管理

### Traffic Manager功能

CARLA的Traffic Manager自动管理NPC车辆：
- 自动导航和路径规划
- 遵守交通规则
- 避让其他车辆
- 保持安全距离

### 自定义NPC行为

在脚本中可以调整Traffic Manager参数：

```python
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # 跟车距离
traffic_manager.set_random_device_seed(0)  # 随机种子（可重现）
```

### 添加更多NPC

```bash
# 生成50辆NPC车辆和30个NPC行人
python spawn_npc_traffic.py -n 50 -w 30
```

---

## Ego Vehicle控制

### 手动控制

使用 `control_ego_vehicle.py` 脚本进行手动控制：

```bash
python control_ego_vehicle.py
```

### 自动控制（自动驾驶）

在 `autonomous_driving_test.py` 中使用 `--autopilot` 参数：

```bash
python autonomous_driving_test.py --autopilot
```

Ego Vehicle将使用Traffic Manager自动导航。

### 程序化控制

在Python脚本中直接控制：

```python
# 创建控制命令
control = carla.VehicleControl(
    throttle=0.5,  # 油门 0.0-1.0
    steer=0.0,    # 转向 -1.0到1.0
    brake=0.0     # 刹车 0.0-1.0
)

# 应用控制
ego_vehicle.apply_control(control)
```

---

## 常见问题

### 1. 无法连接到CARLA服务器

**问题：** 连接超时或失败

**解决方案：**
- 确保CARLA服务器已启动
- 检查端口2000和2001是否被占用
- 检查防火墙设置
- 确认IP地址和端口正确

### 2. NPC车辆不移动

**问题：** NPC车辆生成后静止不动

**解决方案：**
- 确保已调用 `vehicle.set_autopilot(True)`
- 检查Traffic Manager是否已初始化
- 尝试重新生成车辆

### 3. 传感器数据未保存

**问题：** 传感器已安装但数据未保存

**解决方案：**
- 检查数据目录权限
- 确保磁盘空间充足
- 检查传感器是否正确监听：`sensor.listen(callback)`

### 4. 性能问题

**问题：** 运行缓慢或卡顿

**解决方案：**
- 减少NPC数量
- 降低传感器分辨率
- 使用同步模式并调整 `fixed_delta_seconds`
- 关闭不必要的传感器

### 5. 内存不足

**问题：** 内存占用过高

**解决方案：**
- 减少NPC数量
- 降低传感器分辨率
- 定期清理数据目录
- 使用异步模式而非同步模式

### 6. 导入carla模块失败

**问题：** `import carla` 报错

**解决方案：**
```bash
# 重新安装CARLA Python客户端
cd D:\carla\CARLA_0.9.16\PythonAPI\carla\dist
python -m pip install carla-0.9.16-cp312-cp312-win_amd64.whl --force-reinstall
```

---

## 高级用法

### 自定义传感器配置

修改 `autonomous_driving_test.py` 中的 `setup_sensors()` 函数：

```python
# 修改摄像头分辨率
rgb_bp.set_attribute('image_size_x', '3840')
rgb_bp.set_attribute('image_size_y', '2160')

# 修改激光雷达参数
lidar_bp.set_attribute('channels', '128')
lidar_bp.set_attribute('points_per_second', '2000000')
```

### 添加更多传感器

```python
# 添加后视摄像头
rgb_rear_bp = blueprint_library.find('sensor.camera.rgb')
rgb_rear_transform = carla.Transform(
    carla.Location(x=-2.5, z=0.7),
    carla.Rotation(yaw=180)
)
rgb_rear_camera = world.spawn_actor(
    rgb_rear_bp, 
    rgb_rear_transform, 
    attach_to=ego_vehicle
)
```

### 数据后处理

传感器数据可以实时处理：

```python
def process_image(image):
    # 转换为numpy数组
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 移除alpha通道
    
    # 进行图像处理
    # ...
    
    # 保存
    image.save_to_disk('output/image.png')
```

---

## 下一步

1. **学习CARLA API**: 查看 [CARLA Python API文档](https://carla.readthedocs.io/en/latest/python_api/)
2. **探索示例**: 查看 `D:\carla\CARLA_0.9.16\PythonAPI\examples\`
3. **构建UE5版本**: 如需使用UE5版本，参考 `C:\CARLA\carla-ue5-dev\Docs\build_windows_ue5.md`
4. **集成算法**: 将您的自动驾驶或多模态算法集成到测试脚本中

---

## 参考资源

- [CARLA官方文档](https://carla.readthedocs.io/)
- [CARLA GitHub](https://github.com/carla-simulator/carla)
- [CARLA官网](https://carla.org/)
- [First Steps教程](https://carla.readthedocs.io/en/latest/tuto_first_steps/)

---

**安装完成时间**: 2024年  
**CARLA版本**: 0.9.16 (UE4)  
**Python版本**: 3.12.7


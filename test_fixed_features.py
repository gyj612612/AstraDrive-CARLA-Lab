#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的功能：
1. NPC车辆自动运行
2. 多模态数据采集
3. IO处理功能
"""

import carla
import random
import time
import os

print("=" * 60)
print("Testing Fixed Features")
print("=" * 60)

# 连接
print("\n[1/6] Connecting to CARLA server...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("[OK] Connected!")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    exit(1)

# 设置同步模式
print("\n[2/6] Setting up synchronous mode...")
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
traffic_manager.set_random_device_seed(0)
print("[OK] Synchronous mode enabled")

# 生成NPC车辆
print("\n[3/6] Spawning NPC vehicles...")
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
vehicle_blueprints = blueprint_library.filter('vehicle.*')
vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]

npc_vehicles = []
for i in range(10):  # 测试用10辆车
    spawn_point = random.choice(spawn_points)
    vehicle_bp = random.choice(vehicle_blueprints)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        npc_vehicles.append(vehicle)

print(f"[OK] Spawned {len(npc_vehicles)} NPC vehicles")

# 启用自动驾驶
print("\n[4/6] Enabling autopilot for NPC vehicles...")
tm_port = traffic_manager.get_port()
print(f"Traffic Manager port: {tm_port}")

autopilot_count = 0
for vehicle in npc_vehicles:
    try:
        vehicle.set_autopilot(True, tm_port)
        autopilot_count += 1
    except Exception as e:
        print(f"[WARNING] Failed to set autopilot for vehicle {vehicle.id}: {e}")

print(f"[OK] {autopilot_count}/{len(npc_vehicles)} vehicles set to autopilot")

# 等待车辆开始移动
print("\n[5/6] Waiting for vehicles to start moving...")
for i in range(20):  # 等待20帧
    world.tick()

# 检查车辆移动状态
print("\nChecking vehicle movement status...")
moving_count = 0
for vehicle in npc_vehicles:
    velocity = vehicle.get_velocity()
    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
    if speed > 0.1:  # 速度大于0.1 m/s认为在移动
        moving_count += 1
        print(f"  Vehicle {vehicle.id}: Moving at {speed*3.6:.1f} km/h")

print(f"[RESULT] {moving_count}/{len(npc_vehicles)} vehicles are moving")

# 创建Ego Vehicle并安装传感器
print("\n[6/6] Creating Ego Vehicle and installing sensors...")
ego_bp = blueprint_library.filter('vehicle.*')[0]
ego_bp.set_attribute('role_name', 'hero')

# 尝试多个生成点避免碰撞
ego_vehicle = None
for attempt in range(10):
    ego_spawn_point = random.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    if ego_vehicle is not None:
        break

if ego_vehicle is None:
    print("[ERROR] Failed to spawn Ego Vehicle after 10 attempts")
    # 清理并退出
    for vehicle in npc_vehicles:
        vehicle.destroy()
    exit(1)

print(f"[OK] Ego Vehicle created: {ego_vehicle.type_id}")

# 安装RGB摄像头测试数据采集
rgb_bp = blueprint_library.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

frame_count = [0]
test_dir = "test_output"
os.makedirs(test_dir, exist_ok=True)

def test_callback(image):
    frame_count[0] += 1
    if frame_count[0] <= 5:
        file_path = os.path.join(test_dir, f"test_{image.frame:06d}.png")
        image.save_to_disk(file_path)
        print(f"  [OK] Saved frame {image.frame}")

rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)
rgb_camera.listen(test_callback)
print("[OK] RGB camera installed")

# 采集数据
print("\nCollecting data (5 seconds)...")
for i in range(100):  # 100帧，约5秒
    world.tick()

# 检查采集的数据
saved_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
print(f"[RESULT] Collected {len(saved_files)} image files")

# 最终检查
print("\n" + "=" * 60)
print("Final Status Check")
print("=" * 60)

# 再次检查车辆移动
final_moving = 0
for vehicle in npc_vehicles:
    velocity = vehicle.get_velocity()
    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
    if speed > 0.1:
        final_moving += 1

print(f"NPC vehicles moving: {final_moving}/{len(npc_vehicles)}")
print(f"Data collected: {len(saved_files)} files")

# 清理
print("\n[Cleanup] Destroying actors...")
rgb_camera.destroy()
ego_vehicle.destroy()
for vehicle in npc_vehicles:
    vehicle.destroy()

# 恢复异步模式
settings.synchronous_mode = False
world.apply_settings(settings)
traffic_manager.set_synchronous_mode(False)

print("\n" + "=" * 60)
if final_moving > 0 and len(saved_files) > 0:
    print("[SUCCESS] All tests passed!")
    print("  - NPC vehicles are moving automatically")
    print("  - Data collection is working")
else:
    print("[WARNING] Some tests may have issues")
    if final_moving == 0:
        print("  - NPC vehicles are not moving")
    if len(saved_files) == 0:
        print("  - No data collected")
print("=" * 60)


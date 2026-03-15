#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试修复后的脚本 - 验证NPC车辆自动运行和数据采集"""
import carla
import random
import time
import os

print("=" * 60)
print("Testing Fixed Script - NPC Auto-drive & Data Collection")
print("=" * 60)

# 连接
print("\n[1/6] Connecting to CARLA...")
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
print("[OK] Connected!")

# 设置同步模式
print("\n[2/6] Setting up synchronous mode...")
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
tm_port = traffic_manager.get_port()
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
traffic_manager.set_random_device_seed(0)
print(f"[OK] Synchronous mode enabled, TM port: {tm_port}")

# 生成NPC车辆
print("\n[3/6] Spawning NPC vehicles...")
spawn_points = world.get_map().get_spawn_points()
vehicle_blueprints = blueprint_library.filter('vehicle.*')
vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]

npc_vehicles = []
for i in range(5):  # 测试用5辆车
    spawn_point = random.choice(spawn_points)
    vehicle_bp = random.choice(vehicle_blueprints)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        npc_vehicles.append(vehicle)

print(f"[OK] Spawned {len(npc_vehicles)} NPC vehicles")

# 启用自动驾驶
print("\n[4/6] Enabling autopilot for NPC vehicles...")
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
    if i % 5 == 0:
        moving = sum(1 for v in npc_vehicles if v.get_velocity().length() > 0.1)
        print(f"  Frame {i}: {moving}/{len(npc_vehicles)} vehicles moving")

# 检查最终状态
print("\n[6/6] Checking final status...")
moving_vehicles = []
for vehicle in npc_vehicles:
    velocity = vehicle.get_velocity()
    speed = velocity.length()
    if speed > 0.1:
        moving_vehicles.append(vehicle)
        print(f"  Vehicle {vehicle.id}: Moving at {speed*3.6:.1f} km/h")

print(f"\n[RESULT] {len(moving_vehicles)}/{len(npc_vehicles)} NPC vehicles are moving!")

# 创建Ego Vehicle并测试传感器
print("\n[7/7] Testing Ego Vehicle and sensors...")
ego_spawn_point = random.choice(spawn_points)
ego_bp = blueprint_library.filter('vehicle.*')[0]
ego_bp.set_attribute('role_name', 'hero')
ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
print(f"[OK] Ego Vehicle created: {ego_vehicle.type_id}")

# 安装RGB摄像头测试
rgb_bp = blueprint_library.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)

frame_count = [0]
def test_callback(image):
    frame_count[0] += 1
    if frame_count[0] == 1:
        print(f"[OK] Sensor data received! Frame: {image.frame}")

rgb_camera.listen(test_callback)

# 采集几帧数据
print("Collecting sensor data (10 frames)...")
for i in range(10):
    world.tick()

print(f"[OK] Collected {frame_count[0]} frames")

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
if len(moving_vehicles) > 0:
    print("[SUCCESS] Test PASSED! NPC vehicles are moving!")
else:
    print("[WARNING] NPC vehicles may not be moving yet")
print("=" * 60)

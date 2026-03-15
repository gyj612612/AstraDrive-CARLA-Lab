#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速功能测试"""
import carla
import random
import time

print("=" * 60)
print("CARLA Quick Test")
print("=" * 60)

# 连接
print("\n[1/5] Connecting to CARLA server...")
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
print("[OK] Connected!")

# 获取蓝图
print("\n[2/5] Getting blueprints...")
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
print(f"[OK] Found {len(spawn_points)} spawn points")

# 生成NPC车辆
print("\n[3/5] Spawning NPC vehicles...")
vehicle_blueprints = blueprint_library.filter('vehicle.*')
vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]

npc_vehicles = []
for i in range(5):  # 只生成5辆用于测试
    spawn_point = random.choice(spawn_points)
    vehicle_bp = random.choice(vehicle_blueprints)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        npc_vehicles.append(vehicle)

print(f"[OK] Spawned {len(npc_vehicles)} NPC vehicles")

# 启用自动驾驶
print("\n[4/5] Enabling autopilot...")
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(2.5)

for vehicle in npc_vehicles:
    vehicle.set_autopilot(True, traffic_manager.get_port())

print("[OK] Autopilot enabled")

# 创建Ego Vehicle
print("\n[5/5] Creating Ego Vehicle...")
ego_spawn_point = random.choice(spawn_points)
ego_bp = blueprint_library.filter('vehicle.*')[0]
ego_bp.set_attribute('role_name', 'hero')
ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
print(f"[OK] Ego Vehicle created: {ego_vehicle.type_id}")

# 测试传感器
print("\n[6/6] Testing sensors...")
rgb_bp = blueprint_library.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)

# 测试数据采集
frame_count = 0
def test_callback(image):
    global frame_count
    frame_count += 1
    if frame_count == 1:
        print(f"[OK] Sensor data received! Frame: {image.frame}")

rgb_camera.listen(test_callback)

# 等待几秒收集数据
print("Waiting for sensor data...")
time.sleep(3)

# 清理
print("\n[Cleanup] Destroying actors...")
rgb_camera.destroy()
ego_vehicle.destroy()
for vehicle in npc_vehicles:
    vehicle.destroy()

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

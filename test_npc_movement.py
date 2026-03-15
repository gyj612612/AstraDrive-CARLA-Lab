#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试NPC车辆自动运行功能"""
import carla
import random
import time

print("=" * 60)
print("Testing NPC Vehicle Auto Movement")
print("=" * 60)

# 连接
print("\n[1/5] Connecting to CARLA server...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("[OK] Connected!")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("Please start CARLA server first!")
    exit(1)

# 获取蓝图
print("\n[2/5] Getting blueprints...")
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
vehicle_blueprints = blueprint_library.filter('vehicle.*')
vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
print(f"[OK] Found {len(spawn_points)} spawn points, {len(vehicle_blueprints)} vehicle types")

# 生成NPC车辆
print("\n[3/5] Spawning 10 NPC vehicles...")
npc_vehicles = []
for i in range(10):
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
traffic_manager.set_random_device_seed(0)
tm_port = traffic_manager.get_port()
print(f"[INFO] Traffic Manager port: {tm_port}")

for vehicle in npc_vehicles:
    vehicle.set_autopilot(True, tm_port)

print("[OK] Autopilot enabled for all NPC vehicles")

# 等待并检查移动状态
print("\n[5/5] Checking vehicle movement (15 seconds)...")
print("Waiting for vehicles to start moving...")

time.sleep(2)  # 初始等待

# 检查3次，每次间隔5秒
for check in range(3):
    time.sleep(5)
    moving_count = 0
    total_speed = 0.0
    
    for vehicle in npc_vehicles:
        try:
            velocity = vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            speed_kmh = speed * 3.6
            
            if speed_kmh > 0.5:  # 速度大于0.5 km/h认为在移动
                moving_count += 1
                total_speed += speed_kmh
        except:
            pass
    
    avg_speed = total_speed / moving_count if moving_count > 0 else 0
    print(f"  Check {check+1}: {moving_count}/{len(npc_vehicles)} vehicles moving, avg speed: {avg_speed:.1f} km/h")

# 最终检查
print("\n" + "=" * 60)
print("Final Status Check")
print("=" * 60)

moving_vehicles = []
stationary_vehicles = []

for vehicle in npc_vehicles:
    try:
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        speed_kmh = speed * 3.6
        location = vehicle.get_location()
        
        if speed_kmh > 0.5:
            moving_vehicles.append((vehicle.id, speed_kmh, location))
        else:
            stationary_vehicles.append((vehicle.id, location))
    except Exception as e:
        print(f"  [WARNING] Error checking vehicle {vehicle.id}: {e}")

print(f"\nMoving vehicles: {len(moving_vehicles)}")
for vid, speed, loc in moving_vehicles[:5]:  # 只显示前5个
    print(f"  Vehicle {vid}: {speed:.1f} km/h at ({loc.x:.1f}, {loc.y:.1f})")

if len(stationary_vehicles) > 0:
    print(f"\nStationary vehicles: {len(stationary_vehicles)}")
    for vid, loc in stationary_vehicles[:3]:  # 只显示前3个
        print(f"  Vehicle {vid}: at ({loc.x:.1f}, {loc.y:.1f})")

# 清理
print("\n[Cleanup] Destroying vehicles...")
for vehicle in npc_vehicles:
    vehicle.destroy()

print("\n" + "=" * 60)
if len(moving_vehicles) >= len(npc_vehicles) * 0.7:  # 70%以上在移动认为成功
    print("[SUCCESS] NPC vehicles are moving correctly!")
else:
    print("[WARNING] Some NPC vehicles are not moving")
print("=" * 60)


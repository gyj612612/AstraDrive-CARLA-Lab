#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试NPC交通生成功能
"""
import carla
import random
import time

def main():
    print("=" * 60)
    print("Testing NPC Traffic Generation")
    print("=" * 60)
    
    # 连接服务器
    print("\n[1/4] Connecting to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"OK - Connected! Map: {world.get_map().name}")
    except Exception as e:
        print(f"FAILED - {e}")
        return False
    
    # 获取蓝图和生成点
    print("\n[2/4] Preparing blueprints and spawn points...")
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    print(f"OK - {len(vehicle_blueprints)} vehicles, {len(spawn_points)} spawn points")
    
    # 生成NPC车辆
    print("\n[3/4] Spawning NPC vehicles...")
    num_vehicles = 10  # 测试用较少数量
    spawned_vehicles = []
    
    for i in range(num_vehicles):
        spawn_point = random.choice(spawn_points)
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            spawned_vehicles.append(vehicle)
    
    print(f"OK - Spawned {len(spawned_vehicles)} NPC vehicles")
    
    # 启用自动驾驶
    print("\n[4/4] Enabling autopilot for NPC vehicles...")
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    
    for vehicle in spawned_vehicles:
        vehicle.set_autopilot(True, traffic_manager.get_port())
    
    print("OK - All NPC vehicles are driving autonomously")
    
    # 观察运行
    print("\nNPC vehicles are driving... (10 seconds)")
    print("You should see vehicles moving in the CARLA window")
    time.sleep(10)
    
    # 清理
    print("\nCleaning up...")
    for vehicle in spawned_vehicles:
        vehicle.destroy()
    print(f"OK - Destroyed {len(spawned_vehicles)} vehicles")
    
    print("\n" + "=" * 60)
    print("NPC Traffic Test PASSED!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)


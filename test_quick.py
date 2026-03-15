#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证CARLA环境基本功能
"""
import carla
import time
import random

def main():
    print("=" * 60)
    print("CARLA Quick Test")
    print("=" * 60)
    
    # 连接服务器
    print("\n[1/5] Connecting to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"OK - Connected! Map: {world.get_map().name}")
    except Exception as e:
        print(f"FAILED - {e}")
        print("Please start CARLA server first!")
        return False
    
    # 获取蓝图
    print("\n[2/5] Getting blueprints...")
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter('vehicle.*')
    print(f"OK - Found {len(vehicle_bps)} vehicle blueprints")
    
    # 获取生成点
    print("\n[3/5] Getting spawn points...")
    spawn_points = world.get_map().get_spawn_points()
    print(f"OK - Found {len(spawn_points)} spawn points")
    
    # 生成测试车辆
    print("\n[4/5] Spawning test vehicle...")
    try:
        spawn_point = random.choice(spawn_points)
        vehicle_bp = random.choice(vehicle_bps)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"OK - Spawned: {vehicle.type_id}")
        
        # 测试Traffic Manager
        print("\n[5/5] Testing Traffic Manager...")
        traffic_manager = client.get_trafficmanager()
        vehicle.set_autopilot(True, traffic_manager.get_port())
        print("OK - Vehicle set to autopilot")
        
        # 等待几秒观察
        print("\nVehicle is driving... (5 seconds)")
        time.sleep(5)
        
        # 清理
        print("\nCleaning up...")
        vehicle.destroy()
        print("OK - Test vehicle destroyed")
        
    except Exception as e:
        print(f"FAILED - {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)


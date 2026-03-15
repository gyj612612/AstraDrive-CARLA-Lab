#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试CARLA连接"""
import carla
import sys

def test_connection():
    try:
        print("正在连接CARLA服务器...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        print("[OK] Connection successful!")
        print(f"  Map: {world.get_map().name}")
        print(f"  Spawn points: {len(world.get_map().get_spawn_points())}")
        
        # 测试蓝图库
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter('vehicle.*')
        print(f"  Available vehicles: {len(list(vehicles))} types")
        
        return True
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("\n请确保CARLA服务器正在运行!")
        return False

if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)


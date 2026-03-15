#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple CARLA connection test"""

import carla
import sys

def test_connection():
    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        print("[OK] CARLA server connected successfully")
        print(f"Current map: {world.get_map().name}")
        print(f"Available spawn points: {len(world.get_map().get_spawn_points())}")
        
        # Test blueprint library
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter('vehicle.*')
        print(f"Available vehicles: {len(vehicles)}")
        
        walkers = blueprint_library.filter('walker.*')
        print(f"Available walkers: {len(walkers)}")
        
        sensors = blueprint_library.filter('sensor.*')
        print(f"Available sensors: {len(sensors)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False

if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA NPC交通生成脚本
快速生成NPC车辆和行人，让它们自动运行
"""

import carla
import random
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='生成CARLA NPC交通')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('-n', '--num-vehicles', type=int, default=30, help='NPC车辆数量')
    parser.add_argument('-w', '--num-walkers', type=int, default=20, help='NPC行人数量')
    parser.add_argument('--sync', action='store_true', help='使用同步模式')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CARLA NPC交通生成")
    print("=" * 60)
    
    # 连接到CARLA服务器
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✓ 已连接到CARLA服务器")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
    
    # 设置同步模式
    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        print("✓ 同步模式已启用")
    
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # 生成NPC车辆
    print(f"\n生成 {args.num_vehicles} 辆NPC车辆...")
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    
    spawned_vehicles = []
    for i in range(args.num_vehicles):
        spawn_point = random.choice(spawn_points)
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            spawned_vehicles.append(vehicle)
    
    print(f"✓ 已生成 {len(spawned_vehicles)} 辆NPC车辆")
    
    # 启用车辆自动驾驶
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    
    for vehicle in spawned_vehicles:
        vehicle.set_autopilot(True, traffic_manager.get_port())
    
    print("✓ NPC车辆已设置为自动运行")
    
    # 生成NPC行人
    print(f"\n生成 {args.num_walkers} 个NPC行人...")
    walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
    spawned_walkers = []
    walker_controllers = []
    
    for i in range(args.num_walkers):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        
        if spawn_point.location is not None:
            walker_bp = random.choice(walker_blueprints)
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            
            if walker is not None:
                spawned_walkers.append(walker)
                controller_bp = blueprint_library.find('controller.ai.walker')
                controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
                
                if controller is not None:
                    walker_controllers.append(controller)
    
    print(f"✓ 已生成 {len(spawned_walkers)} 个NPC行人")
    
    # 启用行人自动行走
    for controller in walker_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.5 + random.random())
    
    print("✓ NPC行人已设置为自动行走")
    
    print("\n" + "=" * 60)
    print("NPC交通生成完成！")
    print(f"NPC车辆: {len(spawned_vehicles)} 辆")
    print(f"NPC行人: {len(spawned_walkers)} 个")
    print("\n按Ctrl+C停止...")
    print("=" * 60)
    
    try:
        while True:
            if args.sync:
                world.tick()
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n正在清理...")
        
        for vehicle in spawned_vehicles:
            vehicle.destroy()
        
        for walker in spawned_walkers:
            walker.destroy()
        for controller in walker_controllers:
            controller.stop()
            controller.destroy()
        
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)
        
        print("清理完成！")

if __name__ == '__main__':
    main()


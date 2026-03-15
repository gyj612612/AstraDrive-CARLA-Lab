#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Ego Vehicle控制脚本
手动控制Ego Vehicle，同时采集传感器数据
"""

import carla
import random
import argparse
import time
import os
from queue import Queue

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_w, K_s, K_a, K_d, K_SPACE
except ImportError:
    raise RuntimeError('需要安装pygame: pip install pygame')

sensor_queue = Queue()
DATA_DIR = "ego_vehicle_data"
os.makedirs(DATA_DIR, exist_ok=True)

def sensor_callback(sensor_data, sensor_name):
    """传感器数据回调"""
    frame_id = sensor_data.frame
    file_path = os.path.join(DATA_DIR, f"{sensor_name}_{frame_id:06d}.png")
    sensor_data.save_to_disk(file_path)
    sensor_queue.put((frame_id, sensor_name))

def setup_ego_vehicle_sensors(world, ego_vehicle, blueprint_library):
    """在Ego Vehicle上设置传感器"""
    sensors = []
    
    # RGB摄像头
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '1920')
    rgb_bp.set_attribute('image_size_y', '1080')
    rgb_bp.set_attribute('fov', '110')
    rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)
    rgb_camera.listen(lambda data: sensor_callback(data, 'rgb'))
    sensors.append(rgb_camera)
    
    return sensors

def main():
    parser = argparse.ArgumentParser(description='控制CARLA Ego Vehicle')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CARLA Ego Vehicle控制")
    print("=" * 60)
    
    # 连接到CARLA服务器
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✓ 已连接到CARLA服务器")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
    
    # 创建Ego Vehicle
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_point = random.choice(spawn_points)
    
    ego_bp = blueprint_library.filter('vehicle.*')[0]
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
    print(f"✓ Ego Vehicle已创建: {ego_vehicle.type_id}")
    
    # 安装传感器
    sensors = setup_ego_vehicle_sensors(world, ego_vehicle, blueprint_library)
    print(f"✓ 已安装 {len(sensors)} 个传感器")
    
    # 初始化Pygame用于控制
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Ego Vehicle Control")
    clock = pygame.time.Clock()
    
    # 控制参数
    throttle = 0.0
    steer = 0.0
    brake = 0.0
    
    print("\n控制说明:")
    print("  W/↑: 加速")
    print("  S/↓: 刹车")
    print("  A/←: 左转")
    print("  D/→: 右转")
    print("  ESC: 退出")
    print("\n开始控制...")
    
    try:
        running = True
        while running:
            clock.tick(60)
            
            # 处理键盘输入
            keys = pygame.key.get_pressed()
            
            # 加速
            if keys[K_w] or keys[K_UP]:
                throttle = min(throttle + 0.1, 1.0)
                brake = 0.0
            else:
                throttle = max(throttle - 0.05, 0.0)
            
            # 刹车
            if keys[K_s] or keys[K_DOWN]:
                brake = min(brake + 0.1, 1.0)
                throttle = 0.0
            else:
                brake = max(brake - 0.05, 0.0)
            
            # 转向
            if keys[K_a] or keys[K_LEFT]:
                steer = max(steer - 0.1, -1.0)
            elif keys[K_d] or keys[K_RIGHT]:
                steer = min(steer + 0.1, 1.0)
            else:
                steer = steer * 0.9  # 自动回正
            
            # 应用控制
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
            ego_vehicle.apply_control(control)
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
            
            # 更新显示
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 36)
            text = font.render(f"Throttle: {throttle:.2f} | Steer: {steer:.2f} | Brake: {brake:.2f}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            pygame.display.flip()
            
            # 处理传感器队列
            while not sensor_queue.empty():
                try:
                    frame_id, sensor_name = sensor_queue.get_nowait()
                except:
                    break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n正在清理...")
    
    finally:
        # 清理
        for sensor in sensors:
            sensor.destroy()
        ego_vehicle.destroy()
        pygame.quit()
        print(f"\n数据已保存到: {os.path.abspath(DATA_DIR)}")
        print("清理完成！")

if __name__ == '__main__':
    main()


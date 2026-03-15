#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试数据采集和保存功能"""
import carla
import random
import os
import time

# 创建测试数据目录
TEST_DIR = "test_data"
os.makedirs(TEST_DIR, exist_ok=True)

print("=" * 60)
print("Testing Data Collection")
print("=" * 60)

# 连接
print("\n[1/4] Connecting...")
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 创建Ego Vehicle
print("\n[2/4] Creating Ego Vehicle...")
spawn_points = world.get_map().get_spawn_points()
ego_spawn_point = random.choice(spawn_points)
ego_bp = blueprint_library.filter('vehicle.*')[0]
ego_bp.set_attribute('role_name', 'hero')
ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
print(f"[OK] Ego Vehicle created: {ego_vehicle.type_id}")

# 安装RGB摄像头
print("\n[3/4] Installing RGB camera...")
rgb_bp = blueprint_library.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_bp.set_attribute('fov', '110')
rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

frame_count = [0]
def save_image(image):
    frame_count[0] += 1
    file_path = os.path.join(TEST_DIR, f"test_rgb_{image.frame:06d}.png")
    image.save_to_disk(file_path)
    if frame_count[0] <= 3:
        print(f"  [OK] Saved frame {image.frame}: {file_path}")

rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)
rgb_camera.listen(save_image)
print("[OK] Camera installed and listening")

# 等待采集数据
print("\n[4/4] Collecting data (5 seconds)...")
time.sleep(5)

# 检查保存的文件
saved_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
print(f"\n[RESULT] Saved {len(saved_files)} image files")

if len(saved_files) > 0:
    print("[OK] Data collection test PASSED!")
    print(f"  Sample files: {saved_files[:3]}")
else:
    print("[ERROR] No files saved!")

# 清理
print("\n[Cleanup] Destroying actors...")
rgb_camera.destroy()
ego_vehicle.destroy()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)


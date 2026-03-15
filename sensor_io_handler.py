#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传感器数据IO处理模块
提供数据输入输出、格式转换、数据验证等功能
"""

import os
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

class SensorIOHandler:
    """传感器数据IO处理器"""
    
    def __init__(self, base_dir="carla_data"):
        self.base_dir = base_dir
        self.stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'last_frame': 0})
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'sensors': {},
            'frames': []
        }
    
    def save_image(self, image_data, sensor_name, frame_id, data_type='rgb'):
        """保存图像数据（IO输出）"""
        try:
            if data_type == 'rgb':
                dir_path = os.path.join(self.base_dir, "rgb")
                file_path = os.path.join(dir_path, f"{sensor_name}_{frame_id:06d}.png")
                image_data.save_to_disk(file_path)
                
            elif data_type == 'depth':
                dir_path = os.path.join(self.base_dir, "depth")
                file_path = os.path.join(dir_path, f"{sensor_name}_{frame_id:06d}.png")
                image_data.save_to_disk(file_path, carla.ColorConverter.LogarithmicDepth)
                
            elif data_type == 'semantic':
                dir_path = os.path.join(self.base_dir, "semantic")
                file_path = os.path.join(dir_path, f"{sensor_name}_{frame_id:06d}.png")
                image_data.save_to_disk(file_path, carla.ColorConverter.CityScapesPalette)
            
            self.stats[sensor_name]['count'] += 1
            self.stats[sensor_name]['last_frame'] = frame_id
            
            return True, file_path
        except Exception as e:
            self.stats[sensor_name]['errors'] += 1
            return False, str(e)
    
    def save_lidar(self, lidar_data, sensor_name, frame_id):
        """保存激光雷达数据（IO输出）"""
        try:
            dir_path = os.path.join(self.base_dir, "lidar")
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f"{sensor_name}_{frame_id:06d}.ply")
            lidar_data.save_to_disk(file_path)
            
            self.stats[sensor_name]['count'] += 1
            self.stats[sensor_name]['last_frame'] = frame_id
            
            return True, file_path
        except Exception as e:
            self.stats[sensor_name]['errors'] += 1
            return False, str(e)
    
    def save_radar(self, radar_data, sensor_name, frame_id):
        """保存雷达数据（IO输出）"""
        try:
            dir_path = os.path.join(self.base_dir, "radar")
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f"{sensor_name}_{frame_id:06d}.txt")
            
            # 写入雷达检测数据
            with open(file_path, 'w') as f:
                f.write(f"# Radar data frame {frame_id}\n")
                f.write(f"# Format: velocity azimuth altitude depth\n")
                for detection in radar_data:
                    f.write(f"{detection.velocity} {detection.azimuth} {detection.altitude} {detection.depth}\n")
            
            self.stats[sensor_name]['count'] += 1
            self.stats[sensor_name]['last_frame'] = frame_id
            
            return True, file_path
        except Exception as e:
            self.stats[sensor_name]['errors'] += 1
            return False, str(e)
    
    def load_image(self, file_path):
        """加载图像数据（IO输入）"""
        try:
            import cv2
            image = cv2.imread(file_path)
            return True, image
        except Exception as e:
            return False, str(e)
    
    def load_lidar(self, file_path):
        """加载激光雷达数据（IO输入）"""
        try:
            # 使用open3d加载PLY文件
            import open3d as o3d
            point_cloud = o3d.io.read_point_cloud(file_path)
            return True, point_cloud
        except Exception as e:
            return False, str(e)
    
    def load_radar(self, file_path):
        """加载雷达数据（IO输入）"""
        try:
            detections = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) == 4:
                        detections.append({
                            'velocity': float(parts[0]),
                            'azimuth': float(parts[1]),
                            'altitude': float(parts[2]),
                            'depth': float(parts[3])
                        })
            return True, detections
        except Exception as e:
            return False, str(e)
    
    def convert_image_to_numpy(self, image_data):
        """将CARLA图像数据转换为numpy数组（数据转换）"""
        try:
            array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image_data.height, image_data.width, 4))
            array = array[:, :, :3]  # 移除alpha通道，只保留RGB
            array = array[:, :, ::-1]  # BGR转RGB
            return True, array
        except Exception as e:
            return False, str(e)
    
    def save_metadata(self):
        """保存元数据（IO输出）"""
        try:
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['sensors'] = dict(self.stats)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            return True
        except Exception as e:
            return False, str(e)
    
    def get_stats(self):
        """获取统计信息"""
        return dict(self.stats)
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("传感器数据IO统计")
        print("=" * 60)
        for sensor_name, stats in self.stats.items():
            print(f"  {sensor_name}:")
            print(f"    采集帧数: {stats['count']}")
            print(f"    错误次数: {stats['errors']}")
            print(f"    最后帧号: {stats['last_frame']}")
        print("=" * 60)


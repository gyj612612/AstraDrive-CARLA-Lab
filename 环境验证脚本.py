#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA环境验证脚本
检查CARLA环境是否正确配置
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("1. 检查Python版本...")
    version = sys.version_info
    print(f"   Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 7 <= version.minor <= 12:
        print("   ✓ Python版本符合要求")
        return True
    else:
        print("   ✗ Python版本不符合要求（需要3.7-3.12）")
        return False

def check_carla_module():
    """检查CARLA模块"""
    print("\n2. 检查CARLA模块...")
    try:
        import carla
        print("   ✓ CARLA模块导入成功")
        
        # 尝试获取版本信息
        try:
            print(f"   CARLA模块路径: {carla.__file__}")
        except:
            pass
        
        return True
    except ImportError as e:
        print(f"   ✗ CARLA模块导入失败: {e}")
        print("   请运行: python -m pip install D:\\carla\\CARLA_0.9.16\\PythonAPI\\carla\\dist\\carla-0.9.16-cp312-cp312-win_amd64.whl")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n3. 检查依赖包...")
    required_packages = {
        'numpy': 'numpy',
        'pygame': 'pygame',
        'PIL': 'Pillow',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"   ✓ {package_name} 已安装")
        except ImportError:
            print(f"   ✗ {package_name} 未安装")
            missing.append(package_name)
    
    if missing:
        print(f"\n   请安装缺失的包: python -m pip install {' '.join(missing)}")
        return False
    return True

def check_carla_server():
    """检查CARLA服务器连接"""
    print("\n4. 检查CARLA服务器连接...")
    try:
        import carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        print("   ✓ CARLA服务器连接成功")
        print(f"   当前地图: {world.get_map().name}")
        return True
    except Exception as e:
        print(f"   ✗ CARLA服务器连接失败: {e}")
        print("   请确保CARLA服务器正在运行（运行 启动CARLA服务器.bat）")
        return False

def check_directories():
    """检查目录结构"""
    print("\n5. 检查目录结构...")
    
    # 检查CARLA安装目录
    carla_path = r"D:\carla\CARLA_0.9.16"
    if os.path.exists(carla_path):
        print(f"   ✓ CARLA安装目录存在: {carla_path}")
        if os.path.exists(os.path.join(carla_path, "CarlaUE4.exe")):
            print("   ✓ CarlaUE4.exe 存在")
        else:
            print("   ✗ CarlaUE4.exe 不存在")
            return False
    else:
        print(f"   ✗ CARLA安装目录不存在: {carla_path}")
        return False
    
    # 检查UE5源代码目录
    ue5_path = r"C:\CARLA\carla-ue5-dev"
    if os.path.exists(ue5_path):
        print(f"   ✓ UE5源代码目录存在: {ue5_path}")
    else:
        print(f"   ⚠ UE5源代码目录不存在: {ue5_path}（可选）")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("CARLA环境验证")
    print("=" * 60)
    
    results = []
    results.append(check_python_version())
    results.append(check_carla_module())
    results.append(check_dependencies())
    results.append(check_directories())
    results.append(check_carla_server())
    
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ 所有检查通过 ({passed}/{total})")
        print("\n环境配置正确，可以开始使用CARLA！")
        return 0
    else:
        print(f"✗ 部分检查未通过 ({passed}/{total})")
        print("\n请根据上述提示修复问题后重试。")
        return 1

if __name__ == '__main__':
    sys.exit(main())

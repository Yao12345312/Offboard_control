def test_imports():
    """测试所有必需的导入"""
    try:
        print("测试导入...")
        
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        
        import tensorflow as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
        
        from tensorflow.keras.applications import MobileNetV2
        print("✓ MobileNetV2导入成功")
        
        from tensorflow.keras.preprocessing import image
        print("✓ Keras image preprocessing导入成功")
        
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        print("✓ MobileNetV2预处理函数导入成功")
        
        import os
        print("✓ os模块导入成功")
        
        from sklearn.model_selection import train_test_split
        import sklearn
        print(f"✓ scikit-learn版本: {sklearn.__version__}")
        
        import matplotlib.pyplot as plt
        import matplotlib
        print(f"✓ matplotlib版本: {matplotlib.__version__}")
        
        print("\n所有依赖导入成功！可以运行训练脚本。")
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

if __name__ == "__main__":
    test_imports()
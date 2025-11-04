try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy tidak terinstall")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError:
    print("✗ OpenCV tidak terinstall")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ Scikit-learn tidak terinstall")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib tidak terinstall")

print("\nSemua dependencies siap!")
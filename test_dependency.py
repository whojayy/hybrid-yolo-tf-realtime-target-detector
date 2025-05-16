try:
    import packaging
    print("Packaging imported successfully")
    print(f"Packaging version: {packaging.__version__}")
    print(f"Packaging location: {packaging.__file__}")
except ImportError as e:
    print(f"Error importing packaging: {e}")

try:
    import matplotlib
    print("Matplotlib imported successfully")
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"Error importing matplotlib: {e}")

try:
    from ultralytics import YOLO
    print("Ultralytics YOLO imported successfully")
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
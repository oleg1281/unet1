
StoneDetectorEngine (CPU)
=========================
Build on your PC:

open cmd

1) mkdir unet
   cd unet
   python -m venv venv
   venv\Scripts\activate

2) pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install rasterio numpy scikit-image scipy pyshp pyinstaller

3) Put your model in this folder as:
   unet_model.pth

4) Build exe (Windows):
   pyinstaller --clean --noconfirm stone_detector_engine.py ^
      --name stone_detector ^
      --onedir ^
      --add-data "unet_model.pth;." ^
      --add-data "config.json;."

Copy dist\stone_detector to new PC.

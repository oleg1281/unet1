
import os, json, argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import shapefile  # pyshp

class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(UNet, self).__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_ch, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.final = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def run_unet_on_tif(tif_path, model_path, save_mask, save_shp,
                    patch_size=512, stride=256, threshold=0.01, min_obj_pixels=1, device="cpu"):
    device = torch.device(device)
    model = UNet(in_ch=4, out_ch=1).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        profile = src.profile
        transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None

    profile.update(count=1, dtype="float32")

    out_mask = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    with rasterio.open(tif_path) as src:
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                window = Window(x, y, patch_size, patch_size)
                img = src.read(window=window)
                if img.shape[1] == 0 or img.shape[2] == 0:
                    continue
                h, w = img.shape[1], img.shape[2]
                img = normalize(img)
                img_t = torch.tensor(img[np.newaxis, ...], dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()
                pred = pred[:h, :w]
                out_mask[y:y+h, x:x+w] += pred
                weight[y:y+h, x:x+w] += 1

    out_mask /= (weight + 1e-8)

    os.makedirs(os.path.dirname(save_mask), exist_ok=True)
    with rasterio.open(save_mask, "w", **profile) as dst:
        dst.write(out_mask.astype(np.float32), 1)

    binary = (out_mask > threshold)
    binary = remove_small_objects(binary, min_size=min_obj_pixels)
    binary = binary.astype(np.uint8)

    labels_cc = label(binary, connectivity=2)
    regions = regionprops(labels_cc)

    os.makedirs(os.path.dirname(save_shp), exist_ok=True)
    w = shapefile.Writer(save_shp, shapeType=shapefile.POINT)
    w.field("ID", "N")
    w.field("AREA", "N")

    for idx, region in enumerate(regions, 1):
        y, x = region.centroid
        X, Y = rasterio.transform.xy(transform, y, x)
        w.point(float(X), float(Y))
        w.record(ID=idx, AREA=int(region.area))
    w.close()

    if crs_wkt:
        with open(save_shp.replace(".shp", ".prj"), "w", encoding="utf-8") as f:
            f.write(crs_wkt)

    return len(regions)

def load_config(cfg_path):
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    p = argparse.ArgumentParser(description="StoneDetectorEngine (CPU)")
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="unet_model.pth")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min_obj_pixels", type=int, default=None)
    p.add_argument("--config", default="config.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    patch_size = args.patch_size if args.patch_size is not None else cfg.get("patch_size", 512)
    stride = args.stride if args.stride is not None else cfg.get("stride", 256)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.01)
    min_obj_pixels = args.min_obj_pixels if args.min_obj_pixels is not None else cfg.get("min_obj_pixels", 1)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    save_mask = os.path.join(args.out_dir, f"{base}_mask.tif")
    save_shp = os.path.join(args.out_dir, f"{base}_points.shp")

    n = run_unet_on_tif(args.input, args.model, save_mask, save_shp,
                        patch_size=patch_size, stride=stride,
                        threshold=threshold, min_obj_pixels=min_obj_pixels, device="cpu")
    print(f"OK. stones={n}")
    print(f"mask={save_mask}")
    print(f"shp={save_shp}")

if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count()//2))
    main()

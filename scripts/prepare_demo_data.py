"""
Generate demo imagery programmatically — no external downloads required for offline demo.
Creates:
 - single_image: 3 synthetic RS-like scenes
 - temporal_pairs: T1/T2 with controlled change
 - optical_sar_pairs: optical + simulated SAR
 - invalid_inputs: mismatched dimensions
"""
import os, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import random
random.seed(42)
np.random.seed(42)

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "demo")

def ensure_dirs():
    for sub in ["single_image", "temporal_pairs", "optical_sar_pairs", "invalid_inputs"]:
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)

def synthetic_landscape(w=512, h=512, seed=0, with_water=True, with_urban=True):
    rng = np.random.RandomState(seed)
    img = np.zeros((h,w,3), dtype=np.uint8)
    # Base terrain gradient
    # Forest/green base
    g_base = rng.randint(40, 90, size=(h,w))
    r_base = rng.randint(30, 80, size=(h,w))
    b_base = rng.randint(30, 70, size=(h,w))
    img[:,:,0] = r_base
    img[:,:,1] = g_base
    img[:,:,2] = b_base
    # Add texture via perlin-like noise
    noise = rng.randint(-15,15, size=(h,w,3)).astype(int)
    img = np.clip(img.astype(int) + noise, 0,255).astype(np.uint8)
    # Blur a bit
    img = cv2.GaussianBlur(img, (5,5), 1.0)
    # Add water body (river/lake) as dark blue polygon
    if with_water:
        # random polygon
        cnt = rng.randint(0,w)
        pts = np.array([
            [rng.randint(0,w//3), rng.randint(h//4, 3*h//4)],
            [rng.randint(w//3, 2*w//3), rng.randint(h//3, 2*h//3)],
            [rng.randint(2*w//3, w), rng.randint(h//4, 3*h//4)],
            [rng.randint(2*w//3, w), rng.randint(h//4+30, 3*h//4+30)],
            [rng.randint(w//3, 2*w//3), rng.randint(h//3+30, 2*h//3+30)],
            [rng.randint(0,w//3), rng.randint(h//4+30, 3*h//4+30)],
        ], np.int32)
        cv2.fillPoly(img, [pts], (35, 70, 130))
        # add slight variation
        cv2.polylines(img, [pts], True, (45,85,150), 2)
    if with_urban:
        # Add urban grid as gray rectangles with edges
        for _ in range(rng.randint(8,18)):
            x, y = rng.randint(0,w-60), rng.randint(0,h-60)
            cw, ch = rng.randint(20,50), rng.randint(20,45)
            color = tuple(rng.randint(140,210, size=3).tolist())
            cv2.rectangle(img, (x,y), (x+cw, y+ch), color, -1)
            cv2.rectangle(img, (x,y), (x+cw, y+ch), (100,100,100), 1)
            # inner lines (roads)
            if rng.rand() > 0.5:
                cv2.line(img, (x+cw//2, y), (x+cw//2, y+ch), (80,80,80), 1)
    # Add some agricultural fields (rectangular patches)
    for _ in range(rng.randint(3,6)):
        x, y = rng.randint(0,w-80), rng.randint(0,h-80)
        cw, ch = rng.randint(50,120), rng.randint(40,100)
        col = rng.randint(90,160)
        field = np.array([col-10, col+20, col-30], dtype=int)
        field = np.clip(field,0,255)
        cv2.rectangle(img, (x,y), (x+cw, y+ch), tuple(field.tolist()), -1)
        cv2.rectangle(img, (x,y), (x+cw, y+ch), (70,90,60), 1)
    # Final sharpen
    return img

def simulate_sar_from_optical(optical: np.ndarray, seed=0):
    """Simulate SAR-like image from optical: grayscale + speckle + contrast."""
    rng = np.random.RandomState(seed+100)
    gray = cv2.cvtColor(optical, cv2.COLOR_RGB2GRAY).astype(float)
    # Invert some water to dark (SAR water dark)
    # Add speckle multiplicative noise
    speckle = rng.gamma(shape=1.0, scale=1.0, size=gray.shape)
    # Normalize speckle to mean 1
    speckle = speckle / np.mean(speckle)
    sar = gray * speckle * 0.9 + rng.normal(0, 8, gray.shape)
    # Enhance contrast for built-up (bright scatterers)
    # Where optical has bright gray (urban), boost
    bright = gray > 150
    sar[bright] = sar[bright]*1.15 + 15
    # Water dark regions: keep dark
    dark_blue = (optical[:,:,2].astype(float) > optical[:,:,0].astype(float) + 10) & (gray < 100)
    sar[dark_blue] = sar[dark_blue]*0.45
    sar = np.clip(sar, 0,255).astype(np.uint8)
    # Despeckle a bit? Keep speckle
    # Convert to 3-channel for display (gray)
    sar_rgb = np.stack([sar,sar,sar], axis=2)
    # Add slight color tint to make it look SAR-like
    sar_rgb = sar_rgb.astype(int)
    sar_rgb[:,:,0] = np.clip(sar_rgb[:,:,0]*0.95,0,255)
    sar_rgb[:,:,2] = np.clip(sar_rgb[:,:,2]*1.05,0,255)
    return sar_rgb.astype(np.uint8)

def save_png(arr, path):
    Image.fromarray(arr).save(path)
    print(f"Saved {path} {arr.shape}")

def main():
    ensure_dirs()
    # Single images
    for i, seed in enumerate([1, 7, 13]):
        img = synthetic_landscape(512,512, seed=seed, with_water=(i!=2), with_urban=(i!=0))
        save_png(img, os.path.join(BASE, "single_image", f"scene_{i+1}.png"))
        # also save as tiff for GeoTIFF demo (without georef)
        Image.fromarray(img).save(os.path.join(BASE, "single_image", f"scene_{i+1}.tif"))
        # metadata json
        import json
        with open(os.path.join(BASE, "single_image", f"scene_{i+1}.json"), "w") as f:
            json.dump({"scene": i+1, "seed": seed, "size": [512,512], "description": "Synthetic RS landscape"}, f, indent=2)

    # Temporal pairs: T1 (before) and T2 (after flood + new construction)
    # Pair 1: flood
    base = synthetic_landscape(512,512, seed=42, with_water=True, with_urban=True)
    t1 = base.copy()
    # T2: expand water (flood) + remove some urban, add mud
    t2 = base.copy()
    # Expand water polygon
    # Add large water overlay covering 18% more
    overlay = np.zeros_like(t2)
    pts = np.array([[80,180],[250,140],[420,200],[400,340],[220,380],[60,320]], np.int32)
    cv2.fillPoly(overlay, [pts], (30,60,120))
    # Blend water expansion
    mask = (overlay[:,:,0]>0).astype(float)[:,:,None]
    t2 = (t2.astype(float)*(1-mask*0.85) + overlay.astype(float)*mask*0.85).astype(np.uint8)
    # Add speckle to flood area to indicate SAR water confirm
    save_png(t1, os.path.join(BASE, "temporal_pairs", "flood_T1.png"))
    save_png(t2, os.path.join(BASE, "temporal_pairs", "flood_T2.png"))
    Image.fromarray(t1).save(os.path.join(BASE, "temporal_pairs", "flood_T1.tif"))
    Image.fromarray(t2).save(os.path.join(BASE, "temporal_pairs", "flood_T2.tif"))

    # Pair 2: urban expansion
    b1 = synthetic_landscape(512,512, seed=99, with_water=False, with_urban=True)
    b2 = b1.copy()
    # Add new buildings in northeast
    for (x,y) in [(320,80),(370,90),(330,130),(380,140),(340,180)]:
        cv2.rectangle(b2, (x,y), (x+40, y+30), (190,185,175), -1)
        cv2.rectangle(b2, (x,y), (x+40, y+30), (90,90,90), 1)
    # Clear vegetation patch
    cv2.rectangle(b2, (60,300), (180,400), (165,155,140), -1)
    cv2.rectangle(b2, (60,300), (180,400), (100,100,100), 1)
    save_png(b1, os.path.join(BASE, "temporal_pairs", "urban_T1.png"))
    save_png(b2, os.path.join(BASE, "temporal_pairs", "urban_T2.png"))

    # Pair 3: deforestation
    d1 = synthetic_landscape(512,512, seed=123, with_water=True, with_urban=False)
    # Make d1 greener
    d1[:,:,1] = np.clip(d1[:,:,1].astype(int)+18,0,255).astype(np.uint8)
    d2 = d1.copy()
    # Clear forest in center
    cv2.rectangle(d2, (150,150), (350,350), (175,165,145), -1)
    # Add roads
    cv2.line(d2, (150,250), (350,250), (110,110,110), 3)
    cv2.line(d2, (250,150), (250,350), (110,110,110), 3)
    save_png(d1, os.path.join(BASE, "temporal_pairs", "forest_T1.png"))
    save_png(d2, os.path.join(BASE, "temporal_pairs", "forest_T2.png"))

    # Optical-SAR pairs
    opt1 = synthetic_landscape(512,512, seed=200, with_water=True, with_urban=True)
    sar1 = simulate_sar_from_optical(opt1, seed=200)
    save_png(opt1, os.path.join(BASE, "optical_sar_pairs", "pair1_optical.png"))
    save_png(sar1, os.path.join(BASE, "optical_sar_pairs", "pair1_sar.png"))
    Image.fromarray(opt1).save(os.path.join(BASE, "optical_sar_pairs", "pair1_optical.tif"))
    Image.fromarray(sar1).save(os.path.join(BASE, "optical_sar_pairs", "pair1_sar.tif"))

    opt2 = synthetic_landscape(512,512, seed=210, with_water=True, with_urban=True)
    sar2 = simulate_sar_from_optical(opt2, seed=210)
    save_png(opt2, os.path.join(BASE, "optical_sar_pairs", "pair2_optical.png"))
    save_png(sar2, os.path.join(BASE, "optical_sar_pairs", "pair2_sar.png"))

    # Flood hero: T1, T2, SAR
    # Use flood pair plus SAR of T2
    sar_flood = simulate_sar_from_optical(t2, seed=300)
    save_png(sar_flood, os.path.join(BASE, "optical_sar_pairs", "flood_sar.png"))
    # Also copy T1/T2 to a hero folder concept — we will just document
    import json, shutil
    hero_manifest = {
        "hero_demo": {
            "description": "Flood inundation: T1 (before) + T2 (after) + SAR corroboration",
            "files": ["temporal_pairs/flood_T1.png","temporal_pairs/flood_T2.png","optical_sar_pairs/flood_sar.png"],
            "query": "What changed between the two dates, and can the SAR imagery corroborate the change?",
            "expected": "Extensive water expansion ~8-12% -> SAR dark smooth confirms inundation"
        }
    }
    with open(os.path.join(BASE, "manifest.json"), "w") as f:
        json.dump(hero_manifest, f, indent=2)

    # Invalid inputs: mismatched sizes
    small = synthetic_landscape(128,128, seed=999)
    large = synthetic_landscape(512,512, seed=888)
    save_png(small, os.path.join(BASE, "invalid_inputs", "small_128.png"))
    save_png(large, os.path.join(BASE, "invalid_inputs", "large_512.png"))
    # Also a non-image file
    with open(os.path.join(BASE, "invalid_inputs", "not_an_image.txt"), "w") as f:
        f.write("this is not an image")
    # Corrupt image
    with open(os.path.join(BASE, "invalid_inputs", "corrupt.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n not real")

    print("Demo data generation complete.")
    # List
    for root, dirs, files in os.walk(BASE):
        lvl = root.replace(BASE,"")
        for file in files:
            print(os.path.join(lvl, file))

if __name__ == "__main__":
    main()

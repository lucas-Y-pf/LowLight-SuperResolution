# 文件：batch_compare.py（修复版）
import cv2, numpy as np, glob, tqdm
from pathlib import Path

src_dir   = Path('DIV2K/DIV2K_train_HR')
dark_dir  = Path('data/lowlight_syn')
enh_dir   = Path('results/enlighten')
out_dir   = Path('demo_all')
out_dir.mkdir(exist_ok=True)

for f in tqdm.tqdm(list(src_dir.glob('*.png'))[:5], desc="Compare"):
    orig   = cv2.imread(str(f))
    dark   = cv2.imread(str(dark_dir / f.name))
    enhance= cv2.imread(str(enh_dir / f.name))
    demo   = np.hstack([orig, dark, enhance])
    cv2.imwrite(str(out_dir / f.name), demo)

print("✅ 5 张对比图已生成 → demo_all/")
import cv2
import numpy as np
from pathlib import Path

# 路径（都在项目根目录下）
orig_dir   = Path('DIV2K/DIV2K_train_HR')
dark_dir   = Path('data/lowlight_syn')
bright_dir = Path('results/enlighten')
out_dir    = Path('demo_all')          # 输出文件夹
out_dir.mkdir(exist_ok=True)

# 五张图序号
ids = ['0001', '0002', '0003', '0004', '0005']

for idx in ids:
    orig   = cv2.imread(str(orig_dir   / f'{idx}.png'))
    dark   = cv2.imread(str(dark_dir   / f'{idx}.png'))
    bright = cv2.imread(str(bright_dir / f'{idx}.png'))

    # 横向拼：原图 | 低照度 | 亮图
    demo = np.hstack([orig, dark, bright])
    out_file = out_dir / f'demo_{idx}.png'
    cv2.imwrite(str(out_file), demo)
    print(f'✅ 已生成 {out_file}')

print('🎉 全部对比图完成，请在 demo_all/ 挑选最满意的一张重命名为 demo_before_after.png 贴进 README！')
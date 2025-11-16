# 文件：edge_simulate.py
import cv2, torch, glob, tqdm, time
from pathlib import Path

in_path  = Path('data/lowlight_syn')
out_path = Path('results/enlighten_sim')
out_path.mkdir(exist_ok=True)

# 零硬件模拟（笔记本模拟 AArch64 环境）
net = torch.hub.load_state_dict_from_url(
    'https://github.com/lucas-Y-pf/LowLight-SuperResolution/releases/download/v1.0/enlighten_gan.pth',
    map_location='cpu', file_name='enlighten_gan.pth'
)
net.eval()

for f in tqdm.tqdm(list(in_path.glob('*.png'))[:5], desc="Simulate"):
    img = cv2.imread(str(f))
    img_tensor = torch.from_numpy(img / 255.0).unsqueeze(0).permute(0, 3, 1, 2).float()
    out_tensor = net(img_tensor)
    out_img = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
    cv2.imwrite(str(out_path / f.name), out_img)

print("✅ 边缘端模拟推理完成 → results/enlighten_sim/」
import cv2, torch, os, glob, tqdm
from pathlib import Path

# 极简生成器（与 EnlightenGAN 同结构，已嵌入权重参数）
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(3, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True)
        )
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True)
        )
        self.tail = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, 3, 7),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        return self.tail(x)

# 加载你刚才生成的权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Generator().to(device)
net.load_state_dict(torch.load('checkpoints/enlighten_gan.pth', map_location=device))
net.eval()

# 推理
in_path  = Path(r'D:/projects/LowLight-SuperResolution/data/lowlight_syn')
out_path = Path(r'D:/projects/LowLight-SuperResolution/results/enlighten')
out_path = Path(r'D:/projects/LowLight-SuperResolution/results/enlighten')
out_path.mkdir(exist_ok=True)

with torch.no_grad():
    for img_p in tqdm.tqdm(in_path.glob('*'), desc='Brighten'):
        img = cv2.imread(str(img_p))
        if img is None: continue
        img_tensor = torch.from_numpy(img / 255.0).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
        out_tensor = net(img_tensor)
        out_img = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
        cv2.imwrite(str(out_path / img_p.name), out_img)

print(f'✅ 亮图完成 → {out_path.resolve()}')
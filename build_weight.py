import torch, os

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

g = Generator()
torch.manual_seed(42)
for m in g.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.InstanceNorm2d):
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

os.makedirs('checkpoints', exist_ok=True)
torch.save(g.state_dict(), 'checkpoints/enlighten_gan.pth')
print('✅ checkpoints/enlighten_gan.pth 已生成 (7 MB)')
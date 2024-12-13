import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        
        self.blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(out_channels))
        self.blocks.append(nn.ReLU(inplace=True))

        
        for rate in atrous_rates:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        
        self.blocks.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) + 2 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for block in self.blocks:
            res.append(block(x))
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = resnet18(pretrained=True)

        
        self.low_level_features = nn.Sequential(*list(self.backbone.children())[:4])  
        self.high_level_features = nn.Sequential(*list(self.backbone.children())[:-2])  

        
        self.aspp = ASPP(512, 256, atrous_rates=[6, 12, 18])

        
        self.low_level_project = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        low_level_feat = self.low_level_features(x)  
        high_level_feat = self.high_level_features(low_level_feat)  

        
        aspp_out = self.aspp(high_level_feat)

        
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.shape[2:], mode="bilinear", align_corners=False)

        
        low_level_feat = self.low_level_project(low_level_feat)

        
        concat_features = torch.cat([aspp_out, low_level_feat], dim=1)

        
        out = self.decoder(concat_features)

        
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class LoveDADataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
        
        
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask


if __name__ == "__main__":
    
    num_classes = 7  
    batch_size = 2
    num_epochs = 20
    learning_rate = 0.001

    
    train_image_dir = "mmsegmentation/data/loveDA/img_dir/train"
    train_mask_dir = "mmsegmentation/data/loveDA/ann_dir/train"
    val_image_dir = "mmsegmentation/data/loveDA/img_dir/val"
    val_mask_dir = "mmsegmentation/data/loveDA/ann_dir/val"

    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    train_dataset = LoveDADataset(train_image_dir, train_mask_dir, transform=transform)
    val_dataset = LoveDADataset(val_image_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    model = DeepLabV3Plus(num_classes=num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.long().cuda()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    print("Training Complete.")

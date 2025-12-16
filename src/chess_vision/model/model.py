import torch
import torch.nn as nn
import timm

class SimpleSquareClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Create backbone
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0
        )

        # Let the model compute its real output size
        out_size = self._get_feature_dim()

        # Final classifier
        self.classifier = nn.Linear(out_size, num_classes)

    def _get_feature_dim(self):
        # Runs one dummy input through the backbone to discover feature size
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            return out.shape[1]

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

import torch.nn as nn
import torch.utils.data


class FintunedModel(nn.Module):
    def __init__(self, pretrainedModel, classes):
        super(FintunedModel, self).__init__()
        self.features = pretrainedModel.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 256 * 6 * 6)
        y = self.classifier(f)
        return y

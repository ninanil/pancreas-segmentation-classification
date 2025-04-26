class UNetWithClassificationHead(nn.Module):
    def __init__(self, unet: nn.Module, num_classes: int):
        super().__init__()
        self.unet = unet
        self.num_classes = num_classes
        self.classification_head = None  # lazy init

    def forward(self, x):
        # Run full UNet forward pass
        seg_output = self.unet(x)

        # Get encoder features
        features = self.unet.encoder(x)
        encoder_output = features[-1]  # Last encoder output (C=320)

        # Classification head
        if self.classification_head is None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_output.shape[1],
                num_classes=self.num_classes,
                hidden_dim=128
            ).to(encoder_output.device)

        class_output = self.classification_head(encoder_output)
        return seg_output, class_output
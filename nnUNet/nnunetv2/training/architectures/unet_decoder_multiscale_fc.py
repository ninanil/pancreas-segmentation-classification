class UNetWithClassificationHead(nn.Module):
    def __init__(self, unet: nn.Module, num_classes: int):
        super().__init__()
        self.unet = unet
        self.num_classes = num_classes
        self.classification_heads = nn.ModuleList()
        self.final_fc = None

    def forward(self, x):
        # Run full UNet forward pass
        seg_output = self.unet(x)

        # Extract encoder features manually
        features = self.unet.encoder(x)

        # Manually forward through decoder to access specific decoder outputs
        decoder = self.unet.decoder
        decoder_input = features[-1]
        decoder_outputs = []

        for s in range(len(decoder.stages)):
            x = decoder.transpconvs[s](decoder_input)
            x = torch.cat((x, features[-(s + 2)]), dim=1)
            x = decoder.stages[s](x)

            if s == 0 or s == 3 or s == 6:  # First (C=320), Middle (C=128), Last (C=32)
                decoder_outputs.append(x)

            decoder_input = x

        # Create classification heads if not already initialized
        if len(self.classification_heads) == 0:
            for feat in decoder_outputs:
                self.classification_heads.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool3d(1),
                        nn.Flatten(),
                        nn.Linear(feat.shape[1], 64),
                        nn.LeakyReLU(0.01)
                    ).to(feat.device)
                )
            self.final_fc = nn.Linear(64 * len(self.classification_heads), self.num_classes).to(decoder_outputs[0].device)

        # Apply classification heads to each level
        pooled_outputs = [head(feat) for head, feat in zip(self.classification_heads, decoder_outputs)]
        concat_features = torch.cat(pooled_outputs, dim=1)  # shape [B, 64 * 3]

        class_output = self.final_fc(concat_features)
        return seg_output, class_output

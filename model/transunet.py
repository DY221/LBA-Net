import segmentation_models_pytorch as smp
class TransUNet(smp.Unet):
    def __init__(self):
        super().__init__(encoder_name='vit_tiny_patch16_224', encoder_weights=None, in_channels=3, classes=1)

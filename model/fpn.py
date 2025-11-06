import segmentation_models_pytorch as smp
class FPN(smp.FPN):
    def __init__(self):
        super().__init__(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)

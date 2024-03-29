from unet.predictor import UNetSegmentation, UNetSegmentationOnnx
from unet.utils import *
import fire

def predict(weight: str = './weights/unet-epoch800-loss0.0093.pth', weight_onnx: str = './weights/unet-epoch800-loss0.0093.pth.onnx', 
            data_dir: str = './dataset/brain_mri/', idx: int = 23, mode: str = 'torch', start_feat: int = 32 ):

    if mode=="torch":
        model = UNetSegmentation(weight=weight,)
    elif mode=="onnx":
        model = UNetSegmentationOnnx(weight=weight_onnx)
    else:
        raise Exception("Only torch and onnx mode are supported!")

    images, labels = image_loader(idx =idx,data_dir=data_dir)
    model.predict(images,labels)

if __name__ == '__main__':
    fire.Fire(predict)
import numpy as np
import time
import torch
import PIL
from PIL import Image
from pathlib import Path

import onnx
import onnxruntime as ort

from .datamodule import transform_fn
from .module import *
from .utils import *
from typing import *


# from .utils import show_results

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class UNetSegmentationOnnx(object):
    def __init__(self, weight: str = None, num_classes: int = 1, 
                device: str = 'cpu'):

        self.weight = weight
        self.num_classes = num_classes
        self.device = device
        self.transform = transform_fn(train=False)
        
        self._load_check_model()
        self._init_session()

    @property
    def _providers(self):
        return {"cpu":'CPUExecutionProvider', "cuda": 'CUDAExecutionProvider'}
    
    def _init_session(self):
        if self.weight:
            self.session = ort.InferenceSession(self.weight)
        providers = [self._providers.get(self.device, "cpu")]
        self.session.set_providers(providers)
        
    def _load_check_model(self):
        self.onnx_model = onnx.load(self.weight)
        onnx.checker.check_model(self.onnx_model)
        print('model loaded')
        
    def _to_numpy(self, tensor: torch.Tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy() 
        else:
            return tensor.cpu().numpy()        

        
    def _load_image(self, impath: str):
        impath = Path(impath)
        if impath.exists():
            image = Image.open(impath)
            image.convert("RGB")
            return image
        else:
            raise ValueError()

    def _onnx_predict(self, images):
        sess_input = {self.session.get_inputs()[0].name: images}
        ort_outs = self.session.run(None, sess_input)
        onnx_predict = ort_outs[0]
        print('predicted')
        return onnx_predict
    
    def _preprocess(self, inputs: torch.Tensor):
        inputs = self._to_numpy(inputs)

        return inputs

    def _postprocess(self, prediction:np.ndarray):
        threshold : float = 0.5
        prediction[prediction >= threshold] = 1.
        prediction[prediction <= threshold] = 0.
        return prediction

    def _predict(self, image: torch.Tensor, label: torch.Tensor):
        images = self._preprocess(image)
        # labels = self._preprocess(label.requires_grad_(True))
        predict = self._onnx_predict(images)
        result = self._postprocess(predict)
        return result
    
    def predict(self, image: torch.Tensor, label: torch.Tensor):
        result = self._predict(image, label)
        return result



if __name__ == "__main__":
    weight_onnx = './notebook/unet-epoch800-loss0.0093.pth.onnx'
    # weight = './notebook/unet-epoch800-loss0.0093.pth'
    data_dir = './dataset'
    idx = 1
    images, labels = image_loader(idx =idx,data_dir=data_dir)
    net = UNetSegmentationOnnx(weight=weight_onnx)
    start_time = time.time()

    res = net.predict(images, labels)

    total_time =time.time() - start_time

    unit = "s"
    if total_time<1:
        total_time = float(total_time * 1000)
        unit="ms"
        
    print(f'Result : {res}')
    print(f'Execution Time: {total_time:.0f} {unit}')
    # show_results(img,lbl,res)
    
    
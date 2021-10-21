import numpy as np
import time
import torch

import onnx
import onnxruntime as ort
from typing import *

from unet.module import *
from unet.utils import *



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class UNetSegmentation(object):
    def __init__(self, weight: str = None, start_feat: int = 32,  device: str = 'cpu',):
        super(UNetSegmentation).__init__()
        self.weight = weight
        self.device = device
        
        self.model = UNet(start_feat=start_feat)
        self._init_model()
        
        
    def _init_model(self):
        if self.weight:
            self.state_dict = self._load_weight(self.weight)
            self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
    
    def _load_weight(self, weight: str, key=None) -> OrderedDict:
        state_dict = torch.load(weight, map_location=torch.device(self.device))
        return state_dict

    def _predict(self, image: torch.Tensor, label: torch.Tensor):
        threshold: float = 0.5
        start_time = time.time()
        
        self.model.eval()
        with torch.set_grad_enabled(False):
            preds = self.model(image)
            total_time =time.time() - start_time

            output = preds.detach().cpu().numpy()
            output[output >= threshold] = 1.
            output[output <= threshold] = 0.
        
        show_time(total_time)

        result = show_results(image, label, output)
        return result

        
        
        
    def predict(self, image: torch.Tensor, label: torch.Tensor):
        return self._predict(image, label)
        # return result

    
    



class UNetSegmentationOnnx(object):
    def __init__(self, weight: str = None, num_classes: int = 1, 
                device: str = 'cpu'):

        self.weight = weight
        self.num_classes = num_classes
        self.device = device
        
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
        # print('model loaded')
        
    def _to_numpy(self, tensor: torch.Tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy() 
        else:
            return tensor.cpu().numpy()        

    def _onnx_predict(self, images: np.ndarray):
        start_time = time.time()

        sess_input = {self.session.get_inputs()[0].name: images}
        ort_outs = self.session.run(None, sess_input)
        total_time =time.time() - start_time
        onnx_predict = ort_outs[0]

        show_time(total_time)
        return onnx_predict
    
    def _preprocess(self, input: torch.Tensor):
        inputs = self._to_numpy(input)
        return inputs

    def _postprocess(self, prediction: np.ndarray):
        threshold: float = 0.5
        prediction[prediction >= threshold] = 1.
        prediction[prediction <= threshold] = 0.
        return prediction

    def _predict(self, image: torch.Tensor, label: torch.Tensor):
        images = self._preprocess(image)
        predict = self._onnx_predict(images)
        output = self._postprocess(predict)
        result = show_results(image, label, output)
        return result
    
    def predict(self, image: torch.Tensor, label: torch.Tensor):
        result = self._predict(image, label)
        return result



# if __name__ == "__main__":
    # weight_onnx = './weights/unet-epoch800-loss0.0093.pth.onnx'
    # weight = './weights/unet-epoch800-loss0.0093.pth'

    # data_dir = '/home/gaungalif/Workspace/datasets/brain_mri/'
    # idx = 23
    # images, labels = image_loader(idx =idx,data_dir=data_dir)
    
    # net_onnx = UNetSegmentationOnnx(weight=weight_onnx)
    # res = net_onnx.predict(images, labels)
    
    # net = UNetSegmentation(weight=weight)
    # res = net.predict(images,labels)
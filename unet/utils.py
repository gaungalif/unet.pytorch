import numpy as np
import matplotlib.pyplot as plt
import torch

from unet.datamodule import BrainMRISegmentationDataModule


def image_loader(idx: int = None, data_dir: str = None):
    dataset = BrainMRISegmentationDataModule(data_dir)
    valid_loader = dataset.val_dataloader()
    dataiter = iter(valid_loader)
    for _ in range(idx):
        images, labels = dataiter.next()
    return images, labels

class show_results():
    def __init__(self, image: torch.Tensor, label:torch.Tensor, result: np.ndarray):
        self.image = image
        self.label = label
        self.result = result

        self.clean_result(self.image, self.label, self.result)


    def _cfirst_to_clast(self, inputs):
        """
        Args:
            images: numpy array (N, C, W, H) or (C, W, H)
        return:
            images: numpy array(N, W, H, C) or (W, H, C)
        """
        inputs = np.swapaxes(inputs, -3, -2)
        inputs = np.swapaxes(inputs, -2, -1)
        return inputs
    
    def _to_numpy(self, tensor: torch.Tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy() 
        else:
            return tensor.cpu().numpy()  

    def clean_result(self, image: torch.Tensor, mask: torch.Tensor, result: np.ndarray, figsize=(16,26)):
        images, masks =  self._to_numpy(image), self._to_numpy(mask.requires_grad_())
        images = self._cfirst_to_clast(images)
        masks = self._cfirst_to_clast(masks)
        results = self._cfirst_to_clast(result)

        dsc = list(map(dice_score, masks, results))
        num_images = masks.shape[0]

        plt.figure(figsize=figsize)
        for i in range(1):
            plt.subplot(num_images, 3, 1 + 3 * i)
            plt.imshow(images[i])
            plt.xlabel(f'{dsc[i]}')
            
            plt.subplot(num_images, 3, 2 + 3 * i)
            plt.imshow(results[i])
            plt.xlabel('prediction')
            
            plt.subplot(num_images, 3, 3 + 3 * i)
            plt.imshow(masks[i])
            plt.xlabel('ground-truth')
        plt.show()


def dice_score(y_true, y_pred, smoothing=1e-6):
    """
    Args:
        y_true: numpy array with size (W, H, 1) or (1, W, H)
        y_pred: numpy array with size (W, H, 1) or (1, W, H)
        threshold: float
    return:
        dsc: float
    """   
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()

    return (2. * intersection + smoothing) / (union + smoothing)

class DiceScore():
    def __init__(self, threshold=0.5, smoothing=1e-6):
        self.name = 'DSC'
        self.smoothing = smoothing
        self.target = 'max'
        self.threshold = threshold
        
    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true: numpy array with size (N, W, H, 1) or (N, 1, W, H)
            y_pred: numpy array with size (N, W, H, 1) or (N, 1, W, H)
            threshold: float
        return:
            dsc: float
        """
        
        y_pred[y_pred >= self.threshold] = 1.
        y_pred[y_pred <= self.threshold] = 0.
        
        dscs = np.array(list(map(dice_score, y_true, y_pred, [self.smoothing for _ in range(y_pred.shape[0])])))
        
        return np.mean(dscs)

def show_time(total_time):
    unit = "s"
    if total_time<1:
        total_time = float(total_time * 1000)
        unit="ms"
    print(f'Execution Time: {total_time:.0f} {unit}')
import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from external.FoundationStereo.foundation_stereo import FoundationStereo
from external.FoundationStereo.core.utils.utils import InputPadder

class FoundationStereoWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        state_path = './external/FoundationStereo/state'
        cfg = OmegaConf.load(os.path.join(state_path, 'cfg.yaml'))
        self.args = OmegaConf.create(cfg)
        self.model = FoundationStereo(self.args)
        ckpt = torch.load(os.path.join(state_path, 'model_best_bp2.pth'), weights_only=False)
        self.model = self.model.cuda().eval()
        self.model.load_state_dict(ckpt['model'])
        
    def get_disparity(self, left_images, right_images):
        padder = InputPadder(left_images.shape, divis_by=32, force_square=False)
        left_images_padded, right_images_padded = padder.pad(left_images, right_images)

        with torch.no_grad(), torch.cuda.amp.autocast(True):
            outputs_padded = self.model.true_forward(
                left_images_padded,
                right_images_padded,
                iters=self.args.valid_iters,
                test_mode=True,
                output_mode='disparity'
                )
        
        outputs = padder.unpad(outputs_padded.float())
        
        B, C, H, W = outputs.shape
        _, xx = torch.meshgrid(
            torch.arange(H, device=outputs.device), 
            torch.arange(W, device=outputs.device), 
            indexing='ij'
        )
        xx = xx.unsqueeze(0).unsqueeze(0)
        us_right = xx - outputs
        invalid = us_right < 0
        outputs[invalid] = 0
        
        return outputs
    
    def get_logits(self, left_images, right_images):
        """
        Expects input to be divisible by 32
        """
        B, C, H, W = left_images.shape
        assert H % 32 == 0 and W % 32 == 0
        
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            logits = self.model.true_forward(
                left_images,
                right_images,
                iters=self.args.valid_iters,
                test_mode=True,
                output_mode='logits'
                )
            
        return logits.float()

    
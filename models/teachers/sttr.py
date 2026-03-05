import torch
import torch.nn as nn

from external.STTR.module.sttr import STTR

class STTRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
        args = type('', (), {})()
        args.channel_dim = 128
        args.position_encoding='sine1d_rel'
        args.num_attn_layers=6
        args.nheads=8
        args.regression_head='ot'
        args.context_adjustment_layer='cal'
        args.cal_num_blocks=8
        args.cal_feat_dim=16
        args.cal_expansion_ratio=4
        
        self.model = STTR(args).cuda().eval()
        self.model.load_state_dict(torch.load('./external/STTR/state/sceneflow_pretrained_model.pth.tar')['state_dict'], strict=False)
        
    def forward(self, left_images, right_images):
        """
        left_images: [N,C,H,W]
        right_images: [N,C,H,W]
        """
        
        input_data = NestedTensor(left_images, right_images)
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            outputs = self.model(input_data)
            
        disparity = outputs['disp_pred']
        confidence_map = 1 - outputs['occ_pred']
            
        return disparity, confidence_map

class NestedTensor:
    def __init__(self, left, right):
        """
        left: [N,C,H,W]
        right: [N,C,H,W]
        """
        self.left = left
        self.right = right
        self.disp = None
        self.occ_mask = None
        self.occ_mask_right = None
        self.sampled_cols, self.sampled_rows = self.downsample()
        
    def downsample(self):
        h, w = self.left.shape[2], self.left.shape[3]
        bs = 1
        downsample = 3
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()
        
        return sampled_cols, sampled_rows
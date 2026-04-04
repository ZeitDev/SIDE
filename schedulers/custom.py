from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, MultiStepLR

class WarmupCosineAnnealingLR(SequentialLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, start_factor=0.01, eta_min=1e-6):
        warmup_iters = warmup_epochs * steps_per_epoch
        total_iters = total_epochs * steps_per_epoch

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_iters
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=(total_iters - warmup_iters), 
            eta_min=eta_min
        )

        super().__init__(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_iters]
        )
        
class WarmupMultiStepLR(SequentialLR):
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, steps_per_epoch=100, start_factor=0.01):
        warmup_iters = warmup_epochs * steps_per_epoch

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_iters
        )

        multistep_scheduler = MultiStepLR(
            optimizer, 
            milestones=[m * steps_per_epoch for m in milestones], 
            gamma=gamma
        )

        super().__init__(
            optimizer, 
            schedulers=[warmup_scheduler, multistep_scheduler], 
            milestones=[warmup_iters]
        )
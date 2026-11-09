from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    SequentialLR,
)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LRScheduler:
    if not 0 <= num_warmup_steps < num_training_steps:
        raise ValueError(
            "Warmup steps must be non-negative and less than training steps."
        )
    if num_warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0.0)
    warmup_scheduler = LambdaLR(optimizer, lambda step: step / max(1, num_warmup_steps))
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=0.0
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps],
    )

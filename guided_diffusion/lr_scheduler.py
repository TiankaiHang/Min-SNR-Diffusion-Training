import numpy as np
import math


def cosine_scheduler(base_value, final_value, epochs=-1, niter_per_ep=-1, 
                     warmup_epochs=0, start_warmup_value=0, 
                     warmup_steps=-1, total_steps=-1, logger=None):
    logger_fn = print if logger is None else logger.info
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    logger_fn("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    assert (epochs > 0 and niter_per_ep > 0) or (total_steps > 0)
    
    total_iters = epochs * niter_per_ep
    if total_steps > 0:
        total_iters = total_steps

    assert total_iters > warmup_iters

    iters = np.arange(total_iters - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == total_iters
    return schedule


def update_lr_weightdecay(iter, lr_schedule_values, wd_schedule_values, optimizer):
    if lr_schedule_values is not None or wd_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[iter] # * param_group["lr_scale"]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[iter]


def get_lr_wd(optimizer):
    min_lr = 10.
    max_lr = 0.
    weight_decay_value = 0.
    for group in optimizer.param_groups:
        min_lr = min(min_lr, group["lr"])
        max_lr = max(max_lr, group["lr"])

        if group["weight_decay"] > 0:
            weight_decay_value = group["weight_decay"]

    return min_lr, max_lr, weight_decay_value

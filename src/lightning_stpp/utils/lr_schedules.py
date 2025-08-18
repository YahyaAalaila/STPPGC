import math
def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def lr_warmup_cosine(global_step, warmup_steps, base_learning_rate, train_steps):
    """
    Implements the learning rate schedule with warmup and cosine decay.
    Args:
        global_step (int): Current step in training.
        warmup_steps (int): Number of steps for warmup.
        base_learning_rate (float): Base learning rate.
        train_steps (int): Total number of training steps.
    Returns:
        float: Learning rate for the current step.
    """
    # At the moment, this is only used for the neuralstpp model.
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    return learning_rate
import torch


def custom_fwd(cast_inputs=torch.float16):
    def decorator(fn):
        return fn

    return decorator


def custom_bwd(fn):
    return fn

    return decorator


def custom_bwd(fn):
    """Decorator for custom backward functions."""
    return fn

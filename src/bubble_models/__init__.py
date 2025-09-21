"""
Register bubble model implementations
"""
from typing import Callable, Dict, Iterable
from . collision import collision_event

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_ode(name: str) -> Callable:
    """ 
    Simple Decorator to register ode functions
    Example \n
    ---------
        @register("baseline")
        def ode(t, x, up, cp, num_bubbles): ...
    """
    def wrapper(odefun):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Bubble Model `{name}` has already been registered")
        MODEL_REGISTRY[name] = odefun
        return odefun
    return wrapper


def available_models() -> Iterable[str]:
    return MODEL_REGISTRY.keys()


def load_ode(name: str) -> Callable:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        opts = ", ".join(sorted(available_models()))
        raise ValueError(f"Unknown Bubble Model '{name}'. List of availalbel models: {opts}")
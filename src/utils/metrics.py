from dataclasses import dataclass
from contextvars import ContextVar
from numpy.typing import NDArray

@dataclass
class Metrics:
    """
    Simple dataclass to collect metrics
    """
    time         = []
    rmax         = []
    omega_opt    = []
    iters        = []
    linalg_error = []
    last_stepsize= []

# GPT inspired solution to keep collected metrics
_current_metrics: ContextVar[Metrics | None] = ContextVar("current_metrics", default=None)

def get_metrics() -> Metrics | None:
    return _current_metrics.get()

def collect_metrics(metrics_buffer: NDArray):
    """
    Save data from the temporary buffer
    metrics_buffer:
        0 - timestep
        1 - rmax
        2 - omega_opt
        3 - iter
        4 - linalg_error
        5 - last stepsize
    """
    m = get_metrics()
    if m is not None:
        m.time.append(metrics_buffer[0])
        m.rmax.append(metrics_buffer[1])
        m.omega_opt.append(metrics_buffer[2])
        m.iters.append(int(metrics_buffer[3]))
        m.linalg_error.append(metrics_buffer[4])
        m.last_stepsize.append(metrics_buffer[5])
        

class metrics_run:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.token = None
        self.metrics: Metrics | None = None
    def __enter__(self) -> Metrics | None:
        if not self.enabled:
            return None
        self.metrics = Metrics()
        self.token = _current_metrics.set(self.metrics)
        return self.metrics
    def __exit__(self, exc_type, exc, tb):
        if self.token is not None:
            _current_metrics.reset(self.token)
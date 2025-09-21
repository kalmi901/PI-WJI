""" Project Folders / Paths and Constant values / material properties """

from pathlib import Path

# -- PROJECT FOLDERS --
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DATA = PROJECT_ROOT / "data_repository" / "cluster_data"        # Cluster Are Stored Here
TIME_SERIES_DATA = PROJECT_ROOT / "data_repository" / "time_series"
METRICS_DATA = PROJECT_ROOT / "data_repository" / "metrics"
RUNTIME_DATA = PROJECT_ROOT / "data_repository" / "runtime"

HTML_FIGS = PROJECT_ROOT / "html"


"""Fixed joint-training extension of the verified numerical smoke."""
from focus_joint_probe_backoff import SETTINGS as SMOKE

SETTINGS=dict(SMOKE,steps=800,checkpoint_interval=100,worker_training_stop_seconds=1500)

"""Active model / dataset paths for RTMDet validation pipelines."""

from pathlib import Path

MODEL_NAME = 'rtmdet_nano_merged_gbe_v10_gbe_app_vids_v5'
VAL_DATA_DIRNAME = 'merged_gbe_v10_gbe_app_vids_v5'

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / 'data' / VAL_DATA_DIRNAME
ANN_FILE = DATA_ROOT / 'annotations/person_keypoints_val2017.json'
IMG_PREFIX = DATA_ROOT / 'val2017'

PREDICTIONS_DIR = REPO_ROOT / 'predictions'
OVERLAYS_DIR = REPO_ROOT / 'overlays'
REPORTS_DIR = REPO_ROOT / 'reports'
AP_EXPORTED_REPORT = REPORTS_DIR / 'ap_exported.txt'

MMDEPLOY_ROOT = REPO_ROOT.parent / 'mmdeploy'

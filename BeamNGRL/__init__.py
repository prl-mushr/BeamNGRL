from pathlib import Path
import BeamNGRL.dynamics as dynamics


ROOT_PATH = Path(__file__).parent
DATA_PATH = Path(__file__).parents[1] / 'data'
DATASETS_PATH = Path(__file__).parents[1] / 'data' / 'datasets'
DYN_DATA_CONFIG = Path(dynamics.__file__).parent / 'config' / 'datasets'
DYN_EXP_CONFIG = Path(dynamics.__file__).parent / 'config' / 'experiments'
LOGS_PATH = ROOT_PATH.parent / 'logs'
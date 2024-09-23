import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.acquisition import EI_TP, UCB_TP, POI_TP, optimize_acq
from src.gp import ExactGP
from src.tp import TP_v2, TP_v3
from src.test_functions import SinusoidalSynthetic, BraninHoo, Hartmann6
from src.test_functions import add_noise
from src.utils_bo import DataTransformer, generate_initial_data
from src.utils_experiment import set_logger
from src.utils_agt import get_agt_surrogate


__all__ = [
    "EI_TP",
    "UCB_TP",
    "POI_TP",
    "optimize_acq",
    "ExactGP",
    "TP_v2",
    "TP_v3",
    "SinusoidalSynthetic",
    "BraninHoo",
    "Hartmann6",
    "add_noise",
    "DataTransformer",
    "generate_initial_data",
    "set_logger",
    "get_agt_surrogate",
]
import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.acquisition import EI_TP, UCB_TP  # , POI_TP
from src.gp import ExactGP
from src.tp import TP_v1, TP_v2
from src.test_functions import SinusoidaSynthetic, BraninHoo, Hartmann6
from src.utils_bo import DataTransformer, generate_initial_data
from src.utils_experiment import set_logger


__all__ = [
    "EI_TP",
    "UCB_TP",
    # "POI_TP",
    "ExactGP",
    "TP_v1",
    "TP_v2",
    "SinusoidaSynthetic",
    "BraninHoo",
    "Hartmann6",
    "DataTransformer",
    "generate_initial_data",
    "set_logger",
]

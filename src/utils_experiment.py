import datetime
import logging
import os
import sys


def set_logger(log_filename_base, save_dir):
    # Set up logging
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}_{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )


def search_log_files(
    log_dir: str, keywords: list[str], logic: str = "and"
) -> list[str]:
    if logic not in ["or", "and"]:
        raise ValueError("The logic parameter must be 'or' or 'and'.")

    res_files = sorted(os.listdir(log_dir))

    if logic == "and":
        res_files_filtered = [
            f for f in res_files if all(keyword in f for keyword in keywords)
        ]
    elif logic == "or":
        res_files_filtered = [
            f for f in res_files if any(keyword in f for keyword in keywords)
        ]

    return res_files_filtered
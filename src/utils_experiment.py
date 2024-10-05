import ast
import datetime
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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


def parse_array_string(array_str):
    # Remove extra outer brackets and spaces
    array_str = array_str.strip()
    if array_str.startswith("[[") and array_str.endswith("]]"):
        # Removing the outer brackets to make it a simple list
        array_str = array_str[1:-1]

    # Insert commas between numbers
    array_str = re.sub(r"(?<=\d)\s+(?=\d)", ", ", array_str)

    try:
        # Convert the string to a list using ast.literal_eval
        parsed_array = ast.literal_eval(array_str)
        return parsed_array
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing array: {e}")
        return None


class LogParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.settings = {}
        self.initial_data = {"X_initial": [], "y_initial": []}
        self.bo_data = {"X_new": [], "y_new": [], "Beta": [], "Iteration": []}
        self.objective = None

    def _combine_log_entries(self):
        with open(self.file_path, "r") as file:  # 修正: self.log_file → self.file_path
            lines = file.readlines()

        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "

        combined_lines = []
        current_entry = ""

        for line in lines:
            if re.match(timestamp_pattern, line):
                if current_entry:
                    combined_lines.append(current_entry.strip())
                current_entry = line.strip()
            else:
                current_entry += " " + line.strip()

        if current_entry:
            combined_lines.append(current_entry.strip())

        return combined_lines

    def _parse_settings(self, line):
        settings_str = line.split("settings:")[1].strip()
        try:
            # Parse the settings safely
            self.settings = self._safe_parse_settings(settings_str)

            # More flexible regex to capture the class name of the objective_function
            obj_func_match = re.search(
                r"objective_function':\s*<.*?\.([A-Za-z0-9_]+)\s+object\s+at",
                settings_str,
            )

            if obj_func_match:
                objective_function_name = obj_func_match.group(1)
                print(f"Objective Function: {objective_function_name}")
            else:
                print("Objective Function not found in settings.")

        except SyntaxError as e:
            print(f"Failed to parse settings: {e}")
            self.settings = settings_str

    def _safe_parse_settings(self, settings_str):
        # array と <function>, <class> の部分を文字列に置換
        settings_str = re.sub(
            r"<function\s+\w+\s+at\s+0x[0-9a-fA-F]+>", "'<function>'", settings_str
        )
        settings_str = re.sub(r"<class\s+'\w+\.\w+'>", "'<class>'", settings_str)
        settings_str = re.sub(
            r"<\w+\.\w+\s+object\s+at\s+0x[0-9a-fA-F]+>", "'<object>'", settings_str
        )

        # 辞書を文字列から生成（evalは使わず安全なast.literal_evalを使う）
        settings_dict = ast.literal_eval(settings_str)

        return settings_dict

    def parse_log_file(self):
        combined_lines = self._combine_log_entries()
        current_data = {}

        for line in combined_lines:
            # Parse settings
            if "Start BO with settings:" in line:
                settings_str = re.search(r"Start BO with settings: (.*)", line).group(1)

                try:
                    self._parse_settings(line)
                except Exception as e:
                    print(f"Failed to parse settings: {e}")
                    self.settings = settings_str

            # Parse X_initial and y_initial
            elif "X initial:" in line:
                x_initial_str = re.search(r"X initial: (\[.*?\])", line).group(1)
                self.initial_data["X_initial"].append(
                    self._extract_float_list(x_initial_str)
                )
            elif "y initial:" in line:
                y_initial_str = re.search(r"y initial: (\[.*?\])", line).group(1)
                self.initial_data["y_initial"].append(
                    self._extract_float_list(y_initial_str)
                )

            # Parse Beta, Iteration, X_new, and y_new
            elif "Beta:" in line:
                current_data["Beta"] = float(
                    re.search(r"Beta: ([-+]?\d*\.\d+|\d+)", line).group(1)
                )
            elif "Iteration:" in line:
                current_data["Iteration"] = int(
                    re.search(r"Iteration: (\d+) /", line).group(1)
                )
            elif "X new:" in line:
                current_data["X_new"] = self._extract_float_list(
                    re.search(r"X new: (\[\[.*?\]\])", line).group(1)
                )
            elif "y new:" in line:
                current_data["y_new"] = self._extract_float_list(
                    re.search(r"y new: (\[\[.*?\]\])", line).group(1)
                )

                # Check if all required fields are in `current_data`
                if all(
                    key in current_data
                    for key in ["Beta", "Iteration", "X_new", "y_new"]
                ):
                    # Append the current data to `self.bo_data`
                    self.bo_data["Beta"].append(current_data["Beta"])
                    self.bo_data["Iteration"].append(current_data["Iteration"])
                    self.bo_data["X_new"].append(current_data["X_new"])
                    self.bo_data["y_new"].append(current_data["y_new"])

                    # Clear current_data for the next entry
                    current_data.clear()

        # pandas DataFrame に変換
        self.initial_data = pd.DataFrame(self.initial_data)
        self.bo_data = pd.DataFrame(self.bo_data)

        self.initial_data["X_initial"] = self.initial_data["X_initial"].apply(np.array)
        self.initial_data["y_initial"] = self.initial_data["y_initial"].apply(np.array)
        self.initial_data["y_initial"] = self.initial_data["y_initial"].apply(float)

        self.bo_data["X_new"] = self.bo_data["X_new"].apply(np.array)
        self.bo_data["y_new"] = self.bo_data["y_new"].apply(np.array)
        self.bo_data["y_new"] = self.bo_data["y_new"].apply(float)

    def _extract_float_list(self, array_str):
        # Updated regex pattern to capture numbers in scientific notation as well
        number_pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+"
        numbers = re.findall(number_pattern, array_str)
        return [float(num) for num in numbers]

    def create_combined_dataframe(self):
        beta_init = self.bo_data["Beta"].iloc[0]

        df_init = self.initial_data.copy()
        df_init["X_new"] = df_init["X_initial"]
        df_init["y_new"] = df_init["y_initial"]
        df_init.drop(columns=["X_initial", "y_initial"], inplace=True)
        df_init["y_best"] = df_init["y_new"].min()
        df_init["Beta"] = beta_init
        df_init["Beta:n"] = beta_init / len(df_init)
        df_init["Iteration"] = 0

        df_main = self.bo_data.copy()
        df_main["y_best"] = df_main["y_new"].cummin()
        df_main["y_best"] = np.minimum(df_main["y_best"], df_init["y_best"].iloc[0])
        df_main["Beta:n"] = df_main["Beta"] / (len(df_init) + df_main.index)

        df = pd.concat([df_init, df_main])
        return df


# Create a history plot for bo_data using Plotly
def history_plot(bo_data):
    fig = go.Figure()

    # Plot y_new as blue dots
    fig.add_trace(
        go.Scatter(
            x=bo_data["Iteration"],
            y=bo_data["y_new"],
            mode="markers",
            name="y_new",
            marker=dict(color="blue"),
        )
    )

    # Plot y_best as a blue line
    fig.add_trace(
        go.Scatter(
            x=bo_data["Iteration"],
            y=bo_data["y_best"],
            mode="lines",
            name="y_best",
            line=dict(color="blue"),
        )
    )

    # Highlight points where Beta:n >= 1 with orange dots
    high_beta_mask = bo_data["Beta:n"] >= 1
    fig.add_trace(
        go.Scatter(
            x=bo_data["Iteration"][high_beta_mask],
            y=bo_data["y_new"][high_beta_mask],
            mode="markers",
            name="y_new (Beta:n >= 1)",
            marker=dict(color="orange"),
        )
    )

    # Customize the layout
    fig.update_layout(
        title="Bayesian Optimization History",
        xaxis_title="Iteration",
        yaxis_title="Function Value",
        legend_title="Metrics",
    )

    fig.show()

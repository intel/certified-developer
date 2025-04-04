"""
Module to generate dataset for Predictive Asset Maintenance
"""

import os
import warnings
import argparse
import logging
import time
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab3")


def validate_inputs(size: int, save_path: str) -> None:
    """Validates the command line inputs.

    Args:
        size (int): The size of the dataset to generate.
        save_path (str): The path to save the generated dataset.

    Raises:
        ValueError: If any of the inputs are invalid.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Invalid size. It should be a positive integer.")
    if not isinstance(save_path, str) or not save_path:
        raise ValueError("Invalid save path. It should be a non-empty string.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--size", type=int, required=False, default=25000, help="data size"
)
parser.add_argument(
    "-p",
    "--save_path",
    type=str,
    required=True,
    help="path to the output Parquet file within the safe directory",
)
FLAGS = parser.parse_args()

# Validate inputs
try:
    validate_inputs(FLAGS.size, FLAGS.save_path)
except ValueError as e:
    logger.error(f"Validation error: {e}")
    raise

dsize = FLAGS.size
train_path = FLAGS.save_path
train_path = os.path.abspath(os.path.normpath(os.path.join(SAFE_BASE_DIR, train_path)))

# Ensure train_path is still inside SAFE_BASE_DIR
if not train_path.startswith(os.path.abspath(SAFE_BASE_DIR) + os.sep):
    raise ValueError(f"Path is not within the allowed directory {SAFE_BASE_DIR}")

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(train_path), exist_ok=True)

# Generating our data
start = time.time()
logger.info("Generating data with the size %d", dsize)
np.random.seed(1)
manufacturer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
generation_list = ["Gen1", "Gen2", "Gen3", "Gen4"]
lubrication_type_list = ["LTA", "LTB", "LTC"]
product_assignment_list = ["Gala", "Golden_Delicious", "Granny_Smith"]
data = pd.DataFrame(
    {
        "Age": np.random.choice(range(0, 25), dsize, replace=True),
        "Temperature": np.random.randint(low=50, high=300, size=dsize),
        "Last_Maintenance": np.random.normal(0, 60, size=dsize),
        "Motor_Current": np.random.randint(low=0.00, high=10.00, size=dsize),
        "Manufacturer": np.random.choice(manufacturer_list, dsize, replace=True),
        "Generation": np.random.choice(generation_list, dsize, replace=True),
        "Number_Repairs": np.random.choice(range(0, 50), dsize, replace=True),
        "Lubrication": np.random.choice(lubrication_type_list, dsize, replace=True),
        "Product_Assignment": np.random.choice(
            product_assignment_list, dsize, replace=True
        ),
    }
)

# Generating our target variable Asset_Label
logger.info("Generating our target variable Asset_Label")
data["Asset_Label"] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.99, 0.01])

# When age is 0-5 and over 20 change Asset_Label to 1
logger.info("Creating correlation between our variables and our target variable")
logger.info("When age is 0-5 and over 20 change Asset_Label to 1")
data["Asset_Label"] = np.where(
    ((data.Age > 0) & (data.Age <= 5)) | (data.Age > 20), 1, data.Asset_Label
)

# When Temperature is between 150-300 change Asset_Label to 1
logger.info("When Temperature is between 500-1500 change Asset_Label to 1")
data["Asset_Label"] = np.where(
    (data.Temperature >= 150) & (data.Temperature <= 300), 1, data.Asset_Label
)

# When Manufacturer is A, E, or H change Asset_Label to have  80% 1's
logger.info("When Manufacturer is A, E, or H change Asset_Label to 1")
data["Temp_Var"] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.2, 0.8])
data["Asset_Label"] = np.where(
    (data.Manufacturer == "A")
    | (data.Manufacturer == "E")
    | (data.Manufacturer == "H"),
    data.Temp_Var,
    data.Asset_Label,
)

# When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 1's
logger.info("When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 0's")
data["Temp_Var"] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.5, 0.5])
data["Asset_Label"] = np.where(
    (data.Generation == "Gen1") | (data.Generation == "Gen3"),
    data.Temp_Var,
    data.Asset_Label,
)


# When Product Assignment is Pill B change Asset_Label to have 70% to 1's
logger.info("When District is Pill B change Asset_Label to have 70% to 1's")
data["Temp_Var"] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.3, 0.7])
data["Asset_Label"] = np.where(
    (data.Product_Assignment == "Gala"), data.Temp_Var, data.Asset_Label
)


# When Lubrication is LTC change Asset_Label to have 75% to 1's
logger.info("When Lubrication is LTC change Asset_Label to have 75% to 1's")
data["Temp_Var"] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.25, 0.75])
data["Asset_Label"] = np.where(
    (data.Lubrication == "LTC"), data.Temp_Var, data.Asset_Label
)

data.drop("Temp_Var", axis=1, inplace=True)

Categorical_Variables = pd.get_dummies(
    data[["Manufacturer", "Generation", "Lubrication", "Product_Assignment"]],
    drop_first=False,
)
data = pd.concat([data, Categorical_Variables], axis=1)
data.drop(
    ["Manufacturer", "Generation", "Lubrication", "Product_Assignment"],
    axis=1,
    inplace=True,
)

data = data.astype({"Motor_Current": "float64", "Number_Repairs": "float64"})

etime = time.time() - start
datasize = data.shape
logger.info(
    "=====> Time taken %f secs for data generation for the size of %s", etime, datasize
)

# save data to parquet file
try:
    logger.info("Saving the data to %s ...", train_path)
    data.to_parquet(train_path)
    logger.info("DONE")
except Exception as e:
    logger.error(f"Failed to save data: {e}")
    raise

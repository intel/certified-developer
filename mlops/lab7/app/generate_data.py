"""
Module to generate dataset for Predictive Asset Maintenance
"""

from time import time

from numpy import where
from numpy.random import choice, randint, normal, seed
from pandas import DataFrame, concat, get_dummies

from app.__init__ import getLogger, DATA_SIZE, data_path

logger = getLogger(__name__)


# Generating our data
start = time()
logger.info(f"Generating data with the size {DATA_SIZE}")
seed(1)
manufacturer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
generation_list = ["Gen1", "Gen2", "Gen3", "Gen4"]
lubrication_type_list = ["LTA", "LTB", "LTC"]
product_assignment_list = ["Gala", "Golden_Delicious", "Granny_Smith"]
data = DataFrame(
    {
        "Age": choice(range(0, 25), DATA_SIZE, replace=True),
        "Temperature": randint(low=50, high=300, size=DATA_SIZE),
        "Last_Maintenance": normal(0, 60, size=DATA_SIZE),
        "Motor_Current": randint(low=0.00, high=10.00, size=DATA_SIZE),
        "Manufacturer": choice(manufacturer_list, DATA_SIZE, replace=True),
        "Generation": choice(generation_list, DATA_SIZE, replace=True),
        "Number_Repairs": choice(range(0, 50), DATA_SIZE, replace=True),
        "Lubrication": choice(lubrication_type_list, DATA_SIZE, replace=True),
        "Product_Assignment": choice(product_assignment_list, DATA_SIZE, replace=True),
    }
)

# Generating our target variable Asset_Label
logger.info("Generating our target variable Asset_Label")
data["Asset_Label"] = choice(range(0, 2), DATA_SIZE, replace=True, p=[0.99, 0.01])

# When age is 0-5 and over 20 change Asset_Label to 1
logger.info("Creating correlation between our variables and our target variable")
logger.info("When age is 0-5 and over 20 change Asset_Label to 1")
data["Asset_Label"] = where(
    ((data.Age > 0) & (data.Age <= 5)) | (data.Age > 20), 1, data.Asset_Label
)

# When Temperature is between 150-300 change Asset_Label to 1
logger.info("When Temperature is between 500-1500 change Asset_Label to 1")
data["Asset_Label"] = where(
    (data.Temperature >= 150) & (data.Temperature <= 300), 1, data.Asset_Label
)

# When Manufacturer is A, E, or H change Asset_Label to have  80% 1's
logger.info("When Manufacturer is A, E, or H change Asset_Label to 1")
data["Temp_Var"] = choice(range(0, 2), DATA_SIZE, replace=True, p=[0.2, 0.8])
data["Asset_Label"] = where(
    (data.Manufacturer == "A")
    | (data.Manufacturer == "E")
    | (data.Manufacturer == "H"),
    data.Temp_Var,
    data.Asset_Label,
)

# When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 1's
logger.info("When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 0's")
data["Temp_Var"] = choice(range(0, 2), DATA_SIZE, replace=True, p=[0.5, 0.5])
data["Asset_Label"] = where(
    (data.Generation == "Gen1") | (data.Generation == "Gen3"),
    data.Temp_Var,
    data.Asset_Label,
)


# When Product Assignment is Pill B change Asset_Label to have 70% to 1's
logger.info("When District is Pill B change Asset_Label to have 70% to 1's")
data["Temp_Var"] = choice(range(0, 2), DATA_SIZE, replace=True, p=[0.3, 0.7])
data["Asset_Label"] = where(
    (data.Product_Assignment == "Gala"), data.Temp_Var, data.Asset_Label
)


# When Lubrication is LTC change Asset_Label to have 75% to 1's
logger.info("When Lubrication is LTC change Asset_Label to have 75% to 1's")
data["Temp_Var"] = choice(range(0, 2), DATA_SIZE, replace=True, p=[0.25, 0.75])
data["Asset_Label"] = where(
    (data.Lubrication == "LTC"), data.Temp_Var, data.Asset_Label
)

data.drop("Temp_Var", axis=1, inplace=True)

Categorical_Variables = get_dummies(
    data[["Manufacturer", "Generation", "Lubrication", "Product_Assignment"]],
    drop_first=False,
)
data = concat([data, Categorical_Variables], axis=1)
data.drop(
    ["Manufacturer", "Generation", "Lubrication", "Product_Assignment"],
    axis=1,
    inplace=True,
)

data = data.astype({"Motor_Current": "float64", "Number_Repairs": "float64"})

etime = time() - start
datasize = data.shape
logger.info(
    "=====> Time taken %f secs for data generation for the size of %s", etime, datasize
)

# save data to pickle file
train_path = data_path / "train.pkl"
logger.info(f"Saving the data to {train_path} ...")
data.to_pickle(train_path)
logger.info("DONE")

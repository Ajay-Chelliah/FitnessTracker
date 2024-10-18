import pandas as pd
from glob import glob
from src.logger import logging

# # --------------------------------------------------------------
# # Read single CSV file
# # --------------------------------------------------------------

# single_file_acc = pd.read_csv(
#     "../../notebook/Data/Raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )
# single_file_gyro = pd.read_csv(
#     "../../notebook/Data/Raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )
# # --------------------------------------------------------------
# # List all data in data\raw
# # --------------------------------------------------------------

# files = glob("../../notebook/Data/Raw/*.csv")
# file_len = len(files)

# # --------------------------------------------------------------
# # Extract features from filename
# # --------------------------------------------------------------

# data_path = "../../notebook/Data/Raw\\"
# f = files[0]
# participant = f.split("-")[0].replace(data_path, "")
# label = f.split("-")[1]
# category = f.split("-")[2][:-1]

# df = pd.read_csv(f)

# df["Participant"] = participant
# df["Lable"] = label
# df["Category"] = category

# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------

# acc_df = pd.DataFrame()
# gyro_df = pd.DataFrame()
# acc_set = 1
# gyro_set = 1

# data_path = "../notebook/Data/Raw\\"
# for f in files:
#     participant = f.split("-")[0].replace(data_path, "")
#     label = f.split("-")[1]
#     category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#     df = pd.read_csv(f)
#     df["Participant"] = participant
#     df["Label"] = label
#     df["Category"] = category

#     if "Accelerometer" in f:
#         df["Set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])

#     elif "Gyroscope" in f:
#         df["Set"] = gyro_set
#         gyro_set += 1
#         gyro_df = pd.concat([gyro_df, df])

# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------

# acc_df.info()
# pd.to_datetime(df["epoch (ms)"], unit="ms")
# pd.to_datetime(df["time (01:00)"]).dt.day_of_week

# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]

# del gyro_df["epoch (ms)"]
# del gyro_df["time (01:00)"]
# del gyro_df["elapsed (s)"]

# # --------------------------------------------------------------
# # Turn into function
# # --------------------------------------------------------------

file_path = "../../notebook/Data/Raw\\"


def create_data(files):
    acc_df = pd.DataFrame()  # Empty DataFrame to store the Accelerometer data
    gyro_df = pd.DataFrame()  # Empty DataFrame to store the Gyroscope data
    acc_count = 1  # Counter to find the number of CSV files on acclerometer
    gyro_count = 1  # Counter to find the number of CSV files on gyroscope

    for f in files:
        # Getting the participant, label (Exercise name) , category (Heavy/light)
        participant = f.split("-")[0].replace(file_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("_MetaWear_2019").rstrip("123")

        # Creating a dataframe from the CSV file and then creating new columns for the above variables.
        df = pd.read_csv(f)
        df["Participant"] = participant
        df["Label"] = label
        df["Category"] = category

        # Seperating the CSV files into two dataframes (Accelerometer & Gyroscope)
        if "Accelerometer" in f:
            df["Set"] = acc_count
            acc_count += 1
            acc_df = pd.concat([acc_df, df])

        elif "Gyroscope" in f:
            df["Set"] = gyro_count
            gyro_count += 1
            gyro_df = pd.concat([gyro_df, df])

    # Converting the datatype of 'epoch (ms)' column from object into "Datetime" and the using it as an index.
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    # Removing the not important columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]

    return acc_df, gyro_df  # Returning the two Dataframes


data_path = "../../notebook/Data/Raw/*.csv"
files = glob(data_path)  # Storing all the csv files into a variable
acc_df, gyro_df = create_data(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat(
    [acc_df.iloc[:, :3], gyro_df], axis=1
)  # Merging the two Dataframe by removing the repeating columns like Participant, label, Category.
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "label",
    "category",
    "participant",
    "set",
]  # Changing the columns names.

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# Step 1 : Defined the function sampling which contains the aggreagtion methods which is used to apply on the dataframe for downsampling
# Step 2 : Done downsampling for the first 100 rows to evaluate the time taken for it.
# Step 3 : First We grouped the Whole dataframe into groups with respect to Days. Hence 10 groups are formed (11 - 20) and stored inside a list
# Step 4 : Secondly We iterate over the list and do downsampling for each groups seperately and then concatenate the groups one by one forming a single DataFrame

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resample = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
# Converting the Datatype of the Set column from float to int
data_resample.info()
data_resample["set"] = data_resample["set"].astype("int")
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resample.to_pickle(
    "../../artifacts/data_resample.pkl"
)  # Saving the Data in pickle file format.

# save_object(
#     file_path='../../artifacts/Data_resample.pkl',
#     obj=data_resample
# )
logging.info("Data_resample is Saved in pickle format")

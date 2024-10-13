import pandas as pd
from glob import glob

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
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()
    acc_count = 1
    gyro_count = 1

    for f in files:
        participant = f.split("-")[0].replace(file_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("_MetaWear_2019").rstrip("123")

        df = pd.read_csv(f)
        df["Participant"] = participant
        df["Label"] = label
        df["Category"] = category

        if "Accelerometer" in f:
            df["Set"] = acc_count
            acc_count += 1
            acc_df = pd.concat([acc_df, df])

        elif "Gyroscope" in f:
            df["Set"] = gyro_count
            gyro_count += 1
            gyro_df = pd.concat([gyro_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]

    return acc_df, gyro_df


data_path = "../../notebook/Data/Raw/*.csv"
files = glob(data_path)
acc_df, gyro_df = create_data(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
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
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

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
data_resample.info()
data_resample["set"] = data_resample["set"].astype("int")
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resample.to_pickle("../../artifacts/data_resample.pkl")

# save_object(
#     file_path='../../artifacts/Data_resample.pkl',
#     obj=data_resample
# )

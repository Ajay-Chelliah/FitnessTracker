import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../notebook/Data/Raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyro = pd.read_csv(
    "../notebook/Data/Raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data\raw
# --------------------------------------------------------------

files = glob("../notebook/Data/Raw/*.csv")
file_len = len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../notebook/Data/Raw\\"
f = files[0]
participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2][:-1]

df = pd.read_csv(f)

df["Participant"] = participant
df["Lable"] = label
df["Category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()
acc_set = 1
gyro_set = 1

data_path = "../notebook/Data/Raw\\"
for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["Participant"] = participant
    df["Label"] = label
    df["Category"] = category

    if "Accelerometer" in f:
        df["Set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    elif "Gyroscope" in f:
        df["Set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

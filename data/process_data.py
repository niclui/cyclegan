import numpy as np
import pandas as pd
import random
from glob import glob
import os, shutil
import time
from PIL import Image
import sys

def prepare_image_df(horse_dataset_folder_path, zebra_dataset_folder_path):
    horse_images = glob(horse_dataset_folder_path +  "/*.jpg")
    horse_image_df = pd.DataFrame(horse_images, columns=["image_path"])
    horse_image_df["labels"] = 1

    zebra_images = glob(zebra_dataset_folder_path +  "/*.jpg")
    zebra_image_df = pd.DataFrame(zebra_images, columns=["image_path"])
    zebra_image_df["labels"] = 0

    return horse_image_df, zebra_image_df

if __name__ == '__main__':
    # usage: process_data.py [horse_dataset_folder_path] [zebra_dataset_folder_path] [output_csv_name]
    horse_path = sys.argv[1]
    zebra_path = sys.argv[2]
    output_csv_path = sys.argv[3]

    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    horse_image_df, zebra_image_df = prepare_image_df(horse_path, zebra_path)
    horse_image_df.to_csv(output_csv_path + "horse.csv")
    zebra_image_df.to_csv(output_csv_path + "zebra.csv")
    print("datasets generated!")

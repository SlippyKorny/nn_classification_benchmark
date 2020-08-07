# TODO: Thorough testing before releasing the article
from PIL import Image
import argparse
import pandas as pd
import csv
import os

import time

# Arguments
parser = argparse.ArgumentParser(description='Prepares the CNR-EXT dataset for training a CNN')
parser.add_argument('--output', dest='output_folder_name', action='store_const',
                    default="dataset", const=sum,
                    help='name of the folder to which the data will be output (default: "dataset")')

parser.add_argument('--input', dest='input_folder_name', action='store_const',
                    default="images", const=sum,
                    help='name of the folder with the unedited CNR-EXT dataset images (default: "images")')

parser.add_argument('--crops', dest='crop_csv', action='store_const',
                    default="../crops.csv", const=sum, 
                    help='name of the file with the image crop data (default: "crops.csv")')

parser.add_argument('--number', dest='camera_number', action='store_const',
                    default="5", const=sum,
                    help='number of the target camera (default: 5)')

def crop_and_resize_image(img, x, y, width, height):
    dims_tuple = (x, y, width, height)
    return img.crop(box = dims_tuple).resize((75,75))

def exec_operations(name, output_folder, xstart_arr, ystart_arr, xend_arr, yend_arr, free_index, occupied_index):
    img = Image.open(name)

    for i in range(0, len(xstart_arr)):
        img_crop = crop_and_resize_image(img, xstart_arr[i], ystart_arr[i], xend_arr[i], yend_arr[i])
        img_crop.show()
        print("enter 'stop' below to stop execution")
        is_occupied = ""

        while True:
            is_occupied = input("is it occupied? (y/n):")

            if is_occupied == '' or (is_occupied != "stop" and is_occupied[0] != 'y' and is_occupied[0] != 'n'):
                print("incorrect value entered. try again...")
                continue
            if is_occupied == "stop":
                return (free_index, occupied_index, True)
            if is_occupied == "y":
                is_occupied = True
                break
            else:
                is_occupied = False
                break

        filename = output_folder 
        
        if is_occupied:
            filename = filename + "occupied/" + str(occupied_index) + ".jpg"
            occupied_index = occupied_index + 1
        else:
            filename = filename + "free/" + str(free_index) + ".jpg"
            free_index = free_index + 1

        img_crop.save(filename)
    
    return (free_index, occupied_index, False)
        

def iterate(dir, cam_name, csv_file, output_folder):
    xstart_arr = csv_file.pop("xstart")
    ystart_arr = csv_file.pop("ystart")
    xend_arr = csv_file.pop("xend")
    yend_arr = csv_file.pop("yend")
    free_index = 0
    occupied_index = 0

    for weather_folder in os.scandir(dir):
        print(weather_folder.name + " weather")
        for day_folder in os.scandir(weather_folder.path):
            print(day_folder.name + " day")
            for cam_folder in os.scandir(day_folder.path):
                if cam_folder.name == cam_name:
                    for file in os.scandir(cam_folder.path):
                        (free_index, occupied_index, finish) = exec_operations(file.path, output_folder, xstart_arr, ystart_arr, xend_arr, yend_arr, free_index, occupied_index)
                        if finish:
                            return
    return


# Main
def main():
    args = parser.parse_args()
    crops = pd.read_csv(args.crop_csv)
    iterate(args.input_folder_name, "camera" + args.camera_number, crops, args.output_folder_name)

if __name__ == "__main__":
    main()
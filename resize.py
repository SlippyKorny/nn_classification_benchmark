import argparse
from PIL import Image
import os

# Arguments
parser = argparse.ArgumentParser(description='Checks if the parking spot was taken with the use of the passed model '
                                             'and weights file')

parser.add_argument('--width', dest='width', default="75",
                    help='width of the new image (default: 75)')

parser.add_argument('--height', dest='height', default="75",
                    help='height of the new image (default: 75)')

parser.add_argument('--directory', dest='dataset_directory', default="german_traffic_signs/", #TODO: Change to dataset/
                    help='path to the dataset that should be resized (default: "dataset/")')


# Main
def main():
    args = parser.parse_args()
    dirs = os.listdir(args.dataset_directory)
    for item in dirs:
        item_path = args.dataset_directory+item
        if os.path.isdir(item_path):
            for inner_item in os.listdir(item_path):
                inner_item_path = item_path + '/' + inner_item
                if os.path.isfile(inner_item_path) and ".ppm" in inner_item_path:
                    # print(args.dataset_directory+inner_item)
                    img = Image.open(inner_item_path)
                    # f, e = os.path.splitext(args.dataset_directory + item)
                    # print(f)
                    # imResize.save(f + ' resized.jpg', 'JPEG', quality=90)
                    imgResize = img.resize((int(args.height), int(args.width)))
                    inner_item_path_new = inner_item_path.replace('.ppm', '.jpg')
                    imgResize.save(inner_item_path_new, 'JPEG')
                    os.remove(inner_item_path)



if __name__ == "__main__":
    main()

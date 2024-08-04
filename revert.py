import os
import shutil


def moving_images(start_path, end_path):
    directory = os.fsencode(start_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filepath = f"{start_path}/{filename}"
        if os.path.isdir(filepath):
            moving_images(filepath, end_path)
        elif (filename.endswith(".png") or filename.endswith(".jpg")) and filename.startswith("Screenshot"):
            #move
            shutil.move(filepath, end_path)
def del_folders(start_path):
    directory = os.fsencode(start_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filepath = f"{start_path}/{filename}"
        if os.path.isdir(filepath):
            os.rmdir(filepath)

def main():
    PATH_OF_FOLDERS = input("Enter the path where the sorted screenshot folders are stored\nNOTE: Only provide path to a set of folders created by ScreenshotOrganizer.py: ").strip()
    PATH_OF_SCREENSHOTS = input("Enter the path where the screenshots should be unsorted into: ").strip()
    
    moving_images(PATH_OF_FOLDERS, PATH_OF_SCREENSHOTS)
    del_folders(PATH_OF_FOLDERS)

if __name__ == "__main__":
    main()

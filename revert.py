import os
import shutil
PATH_OF_FOLDERS = '/Users/arjunj/Desktop/ss'
PATH_OF_SCREENSHOTS = '/Users/arjunj/Desktop' 


def reverting(start_path, end_path):
    directory = os.fsencode(start_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filepath = f"{start_path}/{filename}"
        if os.path.isdir(filepath):
            reverting(filepath, end_path)
        elif (filename.endswith(".png") or filename.endswith(".jpg")) and filename.startswith("Screenshot"):
            #move
            shutil.move(filepath, end_path)

def main():
    reverting(PATH_OF_FOLDERS, PATH_OF_SCREENSHOTS)

if __name__ == "__main__":
    main()

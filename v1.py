import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os
import numpy as np
import shutil

SIMILARITY_THRESHOLD = .5
PATH_OF_FOLDERS = '/Users/arjunj/Desktop/ss'
PATH_OF_SCREENSHOTS = '/Users/arjunj/Desktop' 

# Load pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
model = model.to(device)
model.eval()

# Define image pre-processing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_images(image_location):
    directory = os.fsencode(image_location)
    list_of_images = []
    image_paths = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = f"{image_location}/{filename}"
        if (filename.endswith(".png") or filename.endswith(".jpg")) and filename.startswith("Screenshot"): 
            image = Image.open(path).convert('RGB')
            image = preprocess(image)
            image_paths.append(path)
            list_of_images.append(image) 
        else:
            continue

    images = torch.stack(list_of_images).to(device) 
    with torch.no_grad():
        embeddings = model(images)
    
    embeddings_list = embeddings.squeeze().cpu().numpy()

    return {path: embedding for path, embedding in zip(image_paths, embeddings)}

def get_folders(folder_location):
    f_centroid = {}
    folder_location = PATH_OF_FOLDERS
    directory = os.fsencode(folder_location)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        subPath = f"{folder_location}/{filename}"
        if os.path.isdir(subPath):
            image_embeddings = get_images(subPath)
            f_centroid[subPath] = np.mean(image_embeddings.values(), axis=0)
    return f_centroid
def cosine_similarity(e1, e2):
    return nn.functional.cosine_similarity(torch.tensor(e1, device=device), torch.tensor(e2, device=device), dim=0)

def main():
    image_embeddings = get_images(PATH_OF_SCREENSHOTS)
    folder_embeddings = get_folders(PATH_OF_FOLDERS)

    # Compute cosine Similarity
    for image_path in image_embeddings:
        max_folder_path = None
        max_cosine_sim = -1
        for folder_path in folder_embeddings:
            cosine_sim = cosine_similarity(image_embeddings[image_path], folder_embeddings[folder_path])
            if cosine_sim > max_cosine_sim:
                max_folder_path = folder_path
                max_cosine_sim = cosine_sim
        if max_folder_path is None or max_cosine_sim < SIMILARITY_THRESHOLD:
            # create new folder
            count = 1
            new_folder_path = f"{PATH_OF_FOLDERS}/f{count}"
            while os.path.exists(new_folder_path):
                count += 1
                new_folder_path = new_folder_path = f"{PATH_OF_FOLDERS}/f{count}" 
            os.makedirs(new_folder_path)
            shutil.move(image_path, new_folder_path)
        else:
            #move to folder
            shutil.move(image_path, max_folder_path)

if __name__ == "__main__":
    main()


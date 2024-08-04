# Image Organizer using ResNet-50

This project organizes images into folders based on visual similarity using a pre-trained ResNet-50 model.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technical Details](#technical-details)
- [What I Learned](#what-i-learned)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)


## Introduction

Do you ever find your screenshots folder cluttered and disorganized? This project aims to solve that problem by automatically organizing images into folders based on their visual similarity. Using a pre-trained ResNet-50 model, this script analyzes the content of your images and sorts them into appropriate folders, making it easier to find and manage your screenshots.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/image-organizer.git
    cd image-organizer
    ```

2. **Set up a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the script:**
    ```bash
    python ScreenshotOrganizer.py
    ```

2. **Enter the paths when prompted:**
    - Path of folders to organize into.
    - Path of screenshots to be sorted.

## Features

- Uses a pre-trained ResNet-50 model to extract image embeddings.
- Calculates cosine similarity to organize images based on visual similarity.
- Automatically creates new folders for images that do not fit existing categories.

## Technical Details

- **Model:** ResNet-50 pre-trained on ImageNet.
- **Libraries:** PyTorch, torchvision, PIL, numpy.
- **Image Processing:** Resizing, normalizing, and tensor conversion.

## What I Learned

- **Image Processing Techniques:** Gained hands-on experience with image preprocessing techniques such as resizing, normalization, and tensor conversion, essential for preparing images for machine learning models.

- **Deep Learning Models:** Improved understanding of how pre-trained deep learning models, specifically ResNet-50, can be used for feature extraction and how to leverage them for various tasks beyond classification.

- **Feature Extraction and Embeddings:** Learned about generating and using embeddings for images, which involves extracting high-level features that capture the essence of an image's content for similarity comparison.

- **Cosine Similarity for Image Matching:** Implemented and understood the concept of cosine similarity to measure the similarity between image embeddings, which is crucial for organizing images based on visual similarity.

- **Performance Optimization:** Gained insights into optimizing performance by caching embeddings to avoid redundant computations and improve the efficiency of the image organization process.


## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.


## Contact Information

For any questions or suggestions, please contact me at [ajanakiraman7@gatech.edu](mailto:ajanakiraman7@gatech.edu).

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [PIL](https://pillow.readthedocs.io/)
# Brain Tumor Detection

This repository provides a comprehensive solution for detecting brain tumors using deep learning techniques. The project encompasses data preparation, model building, training, and deployment through a user-friendly interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Brain tumors are abnormal growths of cells within the brain that can be life-threatening. Early detection is crucial for effective treatment. This project leverages Convolutional Neural Networks (CNNs) to classify MRI images into categories indicating the presence or absence of a tumor.

## Features

- **Data Preparation**: Scripts to preprocess and augment MRI images for training.
- **Model Building**: Implementation of ResNet architecture tailored for tumor detection.
- **Training Utilities**: Functions to train models with options for saving and loading weights.
- **Deployment**: Integration with Gradio to provide an interactive web-based interface for users to upload MRI images and receive predictions.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/samarthsoni3002/Brain_Tumor_Detection.git
   cd Brain_Tumor_Detection
   ```

2. **Install the required packages**:

   Ensure you have Python installed, then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**:

   Organize your MRI images into appropriate directories and use the `data_preparation.py` script to preprocess the data.

   ```bash
   python data_preparation.py
   ```

2. **Model Training**:

   Utilize the `train.py` script to train the CNN model on your dataset.

   ```bash
   python train.py -e <number_of_epochs>
   ```

3. **Interactive Prediction**:

   Launch the Gradio application to interactively predict brain tumors from MRI images.

   ```bash
   python gradio_app.py
   ```

   This will start a local web server where you can upload images and receive predictions.

## Project Structure

- `brain_tumor_trial.ipynb`: Jupyter notebook for experimentation and model trials.
- `data_creation.py`: Contains dataset class.
- `data_preparation.py`: Handles preprocessing of MRI images.
- `extract_data.py`: Utility to extract and organize data from various sources.
- `gradio_app.py`: Deploys the model using Gradio for interactive predictions.
- `model_builder.py`: Build ResNet model architecture.
- `save_and_load_model_weights.py`: Utilities for saving and loading model weights.
- `train.py`: Script to train the model.
- `train_utils.py`: Additional utilities to support the training process.
- `utils.py`: General utility functions used across the project.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

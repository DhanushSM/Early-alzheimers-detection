# Early Alzheimer's Detection

This project aims to detect early signs of Alzheimer's disease using machine learning models. The web application allows users to upload 2D brain scan images or 3D NIfTI files and receive an analysis regarding potential signs of Alzheimer's.

## Repository Overview

- **Repository Name:** Early-alzheimers-detection
- **Owner:** [DhanushSM](https://github.com/DhanushSM)
- **Language:** Jupyter Notebook
- **Default Branch:** main
- **Visibility:** Public

## Project Structure

The project is organized into several directories and files, each serving a specific purpose in the application. Below is an overview of the project structure:

### Directories and Files:

- **`alzheimer_app/`:** The root directory of the project.
  
  - **`static/`:** Contains static files such as CSS, images, and JavaScript. These files are used for styling and enhancing the user interface.
    - **`css/`:** Directory for Cascading Style Sheets (CSS) files.
    - **`images/`:** Directory for image files.
    - **`js/`:** Directory for JavaScript files.
  
  - **`templates/`:** Contains HTML template files used by Flask to render the web pages.
    - **`index.html`**: The home page template where users can upload their brain scans.
    - **`results.html`**: The results page template that displays the analysis results.
  
  - **`app.py`:** The main Flask application file that handles routing, file uploads, and rendering templates. It includes the logic for processing the uploaded brain scans and generating visualizations.
  
  - **`model.py`:** Contains functions to analyze the uploaded brain scans. This file includes mock functions for demonstration purposes, but it can be extended to include actual model predictions.
  
  - **`requirements.txt`:** Lists all the Python dependencies required for the project. This file ensures that all necessary packages are installed when setting up the project.

### Key Components:

- **Flask Application (`app.py`):** The core of the web application. It manages the web server, handles HTTP requests, processes file uploads, and returns the appropriate HTML pages to the user.
- **Brain Scan Analysis (`model.py`):** This module contains the logic for analyzing the uploaded brain scans. It can handle both 2D images and 3D scans, providing predictions and visualizations based on the input data.
- **HTML Templates (`templates/`):** These files define the structure and layout of the web pages. They are rendered by Flask and displayed to the user in their web browser.
- **Static Files (`static/`):** This directory contains all the static assets (CSS, images, and JavaScript) used to enhance the user interface and improve the user experience.

### How It Works:

1. **File Upload:** Users can upload 2D brain scan images (PNG, JPG, JPEG) or 3D NIfTI files (nii, nii.gz) through the web interface.
2. **Analysis:** The uploaded files are processed by the Flask application, which uses the functions in `model.py` to analyze the scans.
3. **Visualization:** For 2D images, the application generates visualizations of the prediction probabilities and displays them on the results page.
4. **Results Display:** The results page shows the analysis results and provides recommendations based on the predictions.

This structure ensures that the project is organized and modular, making it easier to maintain and extend in the future.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/DhanushSM/Early-alzheimers-detection.git
    cd Early-alzheimers-detection
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application:**
    ```bash
    python app.py
    ```

2. **Open your web browser and navigate to `http://127.0.0.1:5000/`.**

3. **Upload a brain scan image or NIfTI file to receive an analysis.**

## Model Details

The model used for image analysis is a convolutional neural network (CNN) trained to detect various stages of Alzheimer's disease. The repository includes mock analysis functions for demonstration purposes, but these can be replaced with actual model predictions.

### Features

- **File Upload:** Users can upload 2D brain scan images (PNG, JPG, JPEG) or 3D NIfTI files (nii, nii.gz).
- **Prediction Analysis:** The application analyzes the uploaded scans and provides a diagnosis confidence.
- **Visualization:** Generates visualizations of the prediction probabilities.
- **Results:** Displays analysis results and recommendations based on the prediction.

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset used for training the model was obtained from [Dataset Source].
- The project was inspired by various research papers on Alzheimer's detection using machine learning.

For any questions or inquiries, please contact [Your Contact Information].

## Additional Information

To view or contribute to the project, visit the repository on GitHub: [Early-alzheimers-detection](https://github.com/DhanushSM/Early-alzheimers-detection).

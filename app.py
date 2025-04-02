from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.secret_key = 'dhanu'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Mock model and analysis functions
class MockAlzheimerModel:
    def predict(self, x):
        # Mock prediction probabilities
        probs = np.random.rand(4)
        return probs.reshape(1, -1) / probs.sum()


def analyze_image(img):
    img = img.convert('RGB')
    img_array = np.array(img.resize((128, 128))) / 255.0

    # Mock analysis
    model = MockAlzheimerModel()
    preds = model.predict(np.expand_dims(img_array, axis=0))[0]
    pred_class = np.argmax(preds)

    return {
        'probabilities': preds,
        'predicted_class': pred_class,
        'confidence': preds[pred_class],
        'class_name': ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia'][pred_class]
    }


def analyze_3d_scan(filepath):
    # Mock 3D scan analysis
    return {
        'hippocampal_volume': "18% below normal range",
        'ventricle_size': "25% above normal range",
        'cortical_thickness': "Reduced in temporal lobe",
        'predicted_condition': "Mild Cognitive Impairment (Early Alzheimer's)",
        'abnormalities': {
            'hippocampus': True,
            'temporal_lobe': True,
            'frontal_lobe': False
        }
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{unique_id}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Determine if it's a 2D or 3D scan
            if ext in ['nii', 'nii.gz']:
                # For 3D scans
                results = analyze_3d_scan(filepath)
                return render_template('results_3d.html',
                                       filename=filename,
                                       results=results)
            else:
                # For 2D images
                img = Image.open(filepath)
                results = analyze_image(img)

                # Generate visualization
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_plot.png")
                generate_visualization(results, plot_path)

                return render_template('results_2d.html',
                                       filename=filename,
                                       plot_filename=f"{unique_id}_plot.png",
                                       results=results)

    return render_template('index.html')


def generate_visualization(results, output_path):
    # Create a visualization of the results
    plt.figure(figsize=(10, 5))

    # Prediction probabilities
    classes = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']
    probabilities = results['probabilities']

    colors = ['#2ecc71' if i == results['predicted_class'] else '#3498db' for i in range(len(classes))]
    plt.barh(classes, probabilities, color=colors)
    plt.xlim(0, 1)
    plt.title('Diagnosis Confidence')
    plt.xlabel('Probability')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Templates should be in a 'templates' folder
# Create these files:
# templates/index.html
# templates/results_2d.html
# templates/results_3d.html

if __name__ == '__main__':
    app.run(debug=True)
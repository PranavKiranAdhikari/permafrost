🌎 Permafrost Thaw Detection - MLOps Mini Project

This project demonstrates an end-to-end MLOps workflow for detecting permafrost thaw using a Convolutional Neural Network (CNN) model.

🛠️ Features

Data Preprocessing & Augmentation: Resize, normalize, rotate, and flip images.

Model Development: CNN trained on thawing and stable permafrost images.

Model Deployment: Flask API exposing a /predict endpoint.

Frontend Testing: Simple HTML form to upload images and get predictions.

Version Control: GitHub repository with proper .gitignore.

📂 Project Structure

PermafrostDetection/
├── src/
│ └── app.py # Flask API code
├── cnn_model.py # CNN training code
├── test_form.html # HTML frontend for testing
├── .gitignore # Files/folders to ignore in Git
└── logs/ # API prediction logs

🚀 How to Run Locally

Clone the repo:
git clone https://github.com/PranavKiranAdhikari/permafrost.git
cd PermafrostDetection

Create a virtual environment:
python -m venv venv
venv\Scripts\activate (Windows)
# source venv/bin/activate (Mac/Linux)

Install dependencies:
pip install -r requirements.txt

Run the Flask API:
python src/app.py

Test predictions:

Open test_form.html in a browser

Upload a permafrost image

View prediction and confidence

📊 Notes

Predictions are logged in logs/predictions_log.csv.

Dataset is not included due to size. Please download or create your own permafrost images (thawing/stable).

⚡ Future Scope

Dockerize the API for containerized deployment

Use MLflow for experiment tracking and model versioning

Add automated monitoring for prediction trends

📌 Requirements

tensorflow>=2.12, numpy, pandas, matplotlib, flask, Pillow, scikit-learn
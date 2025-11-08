# Importing essential libraries and modules

from flask import Flask, render_template, request
import numpy as np
#import pandas as pd
import os
import requests
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------





#disease_dic= ["Eye Spot","Healthy Leaf","Red Leaf Spot","Redrot","Ring Spot"]



from model_predict_E  import pred_skin_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Skin Lesion Detection'
    return render_template('index.html', title=title) 

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Skin Lesion Detection'

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            return render_template('rust.html', title=title)

        # Process the uploaded file
        img = Image.open(file)
        img.save('output.png')

        # Make the prediction
        prediction,confidence2 = pred_skin_disease("output.png") 
        #prediction = str(disease_dic[prediction])

        print("Prediction result:", prediction)
    # Define details for each disease
        # Define details for each sugarcane disease


        ## Define the class names
        #class_names = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis","Viral Pneumonia"]

# Disease information for Diabetic Retinopathy
        disease_info = {
            "Actinic Keratosis": {
                "cause": "Pre-cancerous skin condition caused by sun exposure.",
                "treatment": "Cryotherapy, laser therapy, or topical medications."
            },
            "Basal Cell Carcinoma": {
                "cause": "Most common type of skin cancer due to prolonged UV exposure.",
                "treatment": "Surgical removal, radiation therapy, or targeted drug therapy."
            },
            "Dermatofibroma": {
                "cause": "Benign skin growth caused by fibrous tissue overgrowth.",
                "treatment": "Usually harmless; surgical removal if necessary."
            },
            "Melanoma": {
                "cause": "Serious skin cancer that develops from melanocytes, often due to excessive sun exposure.",
                "treatment": "Surgery, immunotherapy, targeted therapy, or chemotherapy."
            },
            "Nevus": {
                "cause": "Common mole formed by a cluster of melanocytes.",
                "treatment": "No treatment required unless changes in size, shape, or color are observed."
            },
            "Pigmented Benign Keratosis": {
                "cause": "Non-cancerous, pigmented skin growth due to aging.",
                "treatment": "Cryotherapy, laser treatment, or removal if bothersome."
            },
            "Seborrheic Keratosis": {
                "cause": "Common non-cancerous skin growth, often in older adults.",
                "treatment": "Usually requires no treatment; can be removed for cosmetic reasons."
            },
            "Squamous Cell Carcinoma": {
                "cause": "Second most common type of skin cancer caused by sun exposure.",
                "treatment": "Surgical removal, radiation therapy, or chemotherapy."
            },
            "Vascular Lesion": {
                "cause": "Abnormal blood vessel formations on the skin.",
                "treatment": "Laser therapy or minor surgery if needed."
            }
        }

        # Fetch the disease details
        details = disease_info.get(prediction, {})
        cause = details.get("cause", "Unknown condition detected.")
        treatment = details.get("treatment", "No treatment information available.")

        # Render the result page with the prediction and details
        return render_template(
            'rust-result.html', 
            prediction=prediction, 
            cause=cause, 
            treatment=treatment, 
            title="Disease Information",
            confidence2=confidence2
        )



    # Default page rendering
    return render_template('rust.html', title=title)  





# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)

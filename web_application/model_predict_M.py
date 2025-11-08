# ==================== IMPORT LIBRARIES ====================

from tensorflow.keras.applications import MobileNet
from tensorflow.keras import models
import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder

import numpy as np

# ==================== CLASS NAMES ====================

class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

mobilenet_base = MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')

x = GlobalAveragePooling2D()(mobilenet_base.output)

x = Dense(128, activation='relu', kernel_regularizer=l1(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu', kernel_regularizer=l1(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(32, activation='relu', kernel_regularizer=l1(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(9, activation='softmax')(x)

model = models.Model(inputs=mobilenet_base.input, outputs=outputs)
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load only the weights
model.load_weights('mobilenet_skin_cancer_model.h5')


print("MobileNet Model loaded successfully!")

# ==================== IMAGE PREPROCESSING ====================

def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Resize to MobileNet input size
    image = cv2.resize(image, (224, 224))

    # Convert to grayscale for enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Create red edge map overlay
    edges_colored = np.zeros_like(image)
    edges_colored[:, :, 2] = edges
    processed_image = cv2.addWeighted(image, 0.8, edges_colored, 0.5, 0)

    # Save processed image (for your UI)
    cv2.imwrite("static/output_image.png", processed_image)

    # Normalize pixel values (0–1)
    processed_image = processed_image / 255.0

    return np.expand_dims(processed_image, axis=0)

# ==================== PREDICTION FUNCTION ====================

def pred_skin_disease(img_path):
    preprocessed_image = preprocess_single_image(img_path)

    # Predict class probabilities
    predictions = model.predict(preprocessed_image)

    # Get label and confidence
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
    confidence = np.max(predictions)

    print(f"Predicted Label: {predicted_label[0]}, Confidence: {confidence * 100:.2f}%")

    return predicted_label[0], (confidence)



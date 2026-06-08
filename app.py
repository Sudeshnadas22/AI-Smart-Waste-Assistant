from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st

model = load_model("waste_classifier_mobilenet.keras")

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

waste_info = {
    "cardboard": "Recyclable ♻️ | Flatten boxes before recycling.",
    "glass": "Recyclable ♻️ | Rinse and place in a glass recycling bin.",
    "metal": "Recyclable ♻️ | Clean cans before recycling.",
    "paper": "Recyclable ♻️ | Keep paper dry and clean.",
    "plastic": "Recyclable ♻️ | Rinse containers before recycling.",
    "trash": "Non-Recyclable ❌ | Dispose in general waste bin."
}

st.title("♻️ AI Smart Waste Assistant")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(uploaded_file, caption="Uploaded Image")

    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = np.max(prediction) * 100

    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Waste Type: {predicted_class}")
    st.info(waste_info[predicted_class])
    st.write(f"Confidence: {confidence:.2f}%")
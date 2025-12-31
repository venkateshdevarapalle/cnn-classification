import streamlit as st
from PIL import Image
import tempfile
from predict import predict_image

st.set_page_config(page_title="Dog–Cat Classifier", page_icon="🐶")

st.title("🐶🐱 Intelligent Dog–Cat Image Classifier")
st.write("Upload an image of a **dog or cat**. Other images will be rejected.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        label, confidence = predict_image(tmp.name)

    st.subheader("Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if "Neither" in label:
        st.error(label)
    else:
        st.success(label)

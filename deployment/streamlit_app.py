from PIL import Image
import streamlit as st

st.title("This is ThomasWinn's DeepFakeCounter Project :sunglasses:")

uploaded_files = st.file_uploader(
    "Please upload a single / multiple picture file", accept_multiple_files=False
)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        print(image.shape())
        st.image(uploaded_file)
        st.divider()

from io import BytesIO

import lightning.pytorch as pl
import numpy as np
from PIL import Image
import streamlit as st
import torch

from src.models.cnn import CIFAKE_CNN

st.title("This is ThomasWinn's DeepFakeCounter Project :sunglasses:")

uploaded_file = st.file_uploader(
    "Please upload a single / multiple picture file", accept_multiple_files=False, type=['png', 'jpg']
)
if uploaded_file is not None:
    # Create PIL object making sure it's in color and resizing to pixel value our dataset contains
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = image.resize(size=(32, 32))
    
    # Convert PIL to np array then normalize
    image_np = np.array(image, dtype='float')
    image_np = image_np.astype(np.float32)
    image_np = image_np / 255
    
    # Reshape tensor
    image_ten = torch.tensor(image_np)
    image_ten = torch.reshape(image_ten, (1, 3, 32, 32))
    
    # Run prediction
    model = CIFAKE_CNN.load_from_checkpoint('models/4_conv_batch_3_linear/epoch=35-valid_loss=50.84.ckpt')
    model.eval()
    
    # Run image through model to get output prediction
    y_hat = model(image_ten)
    y_hat = y_hat.squeeze()
    prediction = (y_hat > 0.5).float()
    
    statement = 'This image is...'
    
    if prediction:
        st.header(statement + 'REAL')
    else:
        st.header(statement + 'FAKE')
        
    st.image(uploaded_file)
    st.divider()

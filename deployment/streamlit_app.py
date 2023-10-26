import streamlit as st

st.title('This is ThomasWinn"s DeepFakeCounter Project :sunglasses:')

uploaded_files = st.file_uploader("Please upload a single / multiple picture file", accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write('filename:', uploaded_file.name)
        st.write(bytes_data)
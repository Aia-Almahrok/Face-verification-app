import streamlit as st
import deepface.DeepFace as DeepFace
from PIL import Image
import numpy as np
import cv2


st.title("Human face verification ")


def verify_img(image1,image2):
    results = DeepFace.verify(image1,image2)
    return results


upload_1 = st.file_uploader("Choose an image 1", type=["png", "jpg", "webp", "jpeg"])
upload_2 = st.file_uploader("Choose an image 2", type=["png", "jpg", "webp", "jpeg"])

if upload_1  is not None  :
    img_1 = Image.open(upload_1)
    img_np_1 = np.array(img_1)
    st.image(img_np_1, channels="RGB", caption="Uploaded image")

    img_2 = Image.open(upload_2)
    img_np_2 = np.array(img_2)
    st.image(img_np_2, channels="RGB", caption="Uploaded image")

    result=verify_img(img_np_1,img_np_2 )
    if result["verified"]==True:

        st.write("They are the same person")
    else:
        st.write("They are not the same person")

    st.success("Image Processed Successfully")

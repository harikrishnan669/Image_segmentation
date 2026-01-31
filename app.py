import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from scipy import ndimage as ndi

st.set_page_config(page_title="Image Restoration & Segmentation", layout="wide")

st.title("Image Restoration and Segmentation")

# ---------------- IMAGE RESTORATION ----------------
st.header("Image Restoration")

restoration_file = st.file_uploader(
    "Upload image for restoration", type=["jpg", "png", "jpeg"], key="rest"
)

if restoration_file is not None:
    file_bytes = np.asarray(bytearray(restoration_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Add Noise
    noise = np.random.normal(0, 25, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Blur
    blurred_img = cv2.GaussianBlur(noisy_img, (7, 7), 0)

    # Denoising
    denoised_img = cv2.fastNlMeansDenoising(blurred_img, None, 10, 7, 21)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    restored_img = cv2.filter2D(denoised_img, -1, kernel)

    st.subheader("Restoration Results")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.image(img, caption="Original", channels="GRAY")
    col2.image(noisy_img, caption="Noisy", channels="GRAY")
    col3.image(blurred_img, caption="Blurred", channels="GRAY")
    col4.image(denoised_img, caption="Denoised", channels="GRAY")
    col5.image(restored_img, caption="Restored", channels="GRAY")

# ---------------- IMAGE SEGMENTATION ----------------
st.header("Image Segmentation")

segmentation_file = st.file_uploader(
    "Upload image for segmentation", type=["jpg", "png", "jpeg"], key="seg"
)

if segmentation_file is not None:
    file_bytes = np.asarray(bytearray(segmentation_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # K-Means Segmentation
    Z = img_color.reshape((-1, 3))
    Z = np.float32(Z)

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    segmented_kmeans = center[label.flatten()]
    segmented_kmeans = segmented_kmeans.reshape(img_color.shape)

    # Watershed Segmentation
    distance = ndi.distance_transform_edt(thresh)
    markers = ndi.label(distance > 0.4 * distance.max())[0]
    labels = watershed(-distance, markers, mask=thresh)

    st.subheader("Segmentation Results")
    col1, col2, col3, col4 = st.columns(4)

    col1.image(img_color, caption="Original", channels="BGR")
    col2.image(thresh, caption="Thresholding", channels="GRAY")
    col3.image(segmented_kmeans, caption="K-Means Segmentation", channels="BGR")
    col4.image(labels, caption="Watershed Segmentation", channels="JET")

st.success("✔ Processing Complete")

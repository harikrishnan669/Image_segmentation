

# Image Restoration and Segmentation using Streamlit

## Objectives

* To apply image processing techniques for **restoring degraded images**
* To perform **meaningful image segmentation**
* To visualize **all intermediate processing stages**
* To implement the solution using **Python scripting**

---

## 🛠️ Technologies Used

* **Python 3**
* **Streamlit** – Web interface
* **OpenCV** – Image processing
* **NumPy** – Numerical operations
* **Matplotlib** – Visualization
* **scikit-image** – Watershed segmentation
* **SciPy** – Distance transform

---

## 📂 Project Structure

```
Image-Restoration-Segmentation/
│
├── app.py   		       # Main Streamlit application
├── README.md                 # Project documentation
├── requirements.txt          # Required libraries (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Install Required Libraries

```bash
pip install streamlit opencv-python numpy matplotlib scikit-image scipy
```

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Image Restoration Module

### Steps Involved

1. Upload grayscale image
2. Add Gaussian noise
3. Apply Gaussian blur
4. Perform denoising using Non-Local Means filter
5. Apply sharpening filter for restoration

### Techniques Used

* Gaussian Noise
* Gaussian Blur
* Non-Local Means Denoising
* Image Sharpening (Spatial Filtering)

### Output

* Original Image
* Noisy Image
* Blurred Image
* Denoised Image
* Restored Image

---

## 🧩 Image Segmentation Module

### Steps Involved

1. Upload color image
2. Convert to grayscale
3. Apply Otsu’s thresholding
4. Perform K-Means clustering
5. Apply Watershed segmentation

### Techniques Used

* Thresholding
* K-Means Clustering
* Watershed Algorithm

### Output

* Original Image
* Thresholded Image
* K-Means Segmented Image
* Watershed Segmented Image

---

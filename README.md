# 🌍 Predicting Land Use & Land Cover (LULC) from Satellite Imagery using ML

This project focuses on identifying land categories from satellite images using a **stacked ensemble deep learning model**. It leverages three CNN backbones — **MobileNet**, **EfficientNet**, and **ResNet50** — combined with a **Logistic Regression meta-classifier** to achieve **97.3% accuracy**. A custom **Streamlit interface** enables users to upload images, perform patch-wise classification, and visualize class distribution with actionable insights.

---

## 🧭 Why This Project?

Land Use and Land Cover (LULC) analysis is crucial for:

- **Environmental Monitoring** – Detect forest loss, vegetation changes, wetland shifts.  
- **Urban Planning** – Track infrastructure, urban sprawl, and zoning shifts.  
- **Disaster Management** – Identify high-risk regions for floods, landslides, fires.

---

## 🏗️ Model Architecture (Stacked Ensemble)
### Architecture Diagram
  <img width="898" height="510" alt="Screenshot 2025-10-12 203610" src="https://github.com/user-attachments/assets/4ed708e8-df59-4db1-ac91-44e8cd621e94" />

### Methodology
  <img width="569" height="544" alt="image" src="https://github.com/user-attachments/assets/24921609-62ea-477e-873e-3fa151efa88f" />

- Three CNN backbones extract features from image patches.  
- Logistic Regression combines predictions (meta-classifier) for final LULC classification.

---

## 🗂 Dataset Used

- **NWPU-RESISC45** – 31,500 aerial images across 45 land categories.  
- Models trained and tested using **Kaggle** and **Google Colab** (credits acknowledged).

---

## 🖥️ Streamlit Application Features

✔ Upload satellite images  
✔ Patch-wise classification (tile-based)  
✔ Pie chart visualization of class distribution  
✔ Knowledge-based suggestions for detected classes  
✔ Export results as CSV  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Keerthi132/Predicting-Land-Use-Land-Cover-from-Satellite-Imagery-using-ML.git
cd Predicting-Land-Use-Land-Cover-from-Satellite-Imagery-using-ML

# Install dependencies
pip install -r requirements.txt

# Place model weights
# (Download .pth and .pkl files and put them inside /models/ directory)

# Run the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
│── app.py                # Main Streamlit application  
│── models/               # Folder to store downloaded model weights  
│── README.md             # Documentation  
│── requirements.txt      # Dependencies  
│── .gitignore            # Git config ignore file
```

---

## 🎯 Future Enhancements

- Add segmentation for pixel-level LULC mapping  
- Integrate Shapefile / GeoTIFF support  
- Deploy as full web service (API + UI)

---

## 🙏 Acknowledgements

- **Dataset:** NWPU-RESISC45  
- **Training Platforms:** Kaggle & Google Colab  
- Inspired by real-world applications in remote sensing, GIS, and environmental AI

---

## 🗒️ License

This project is for academic and research use. Not for commercial deployment.

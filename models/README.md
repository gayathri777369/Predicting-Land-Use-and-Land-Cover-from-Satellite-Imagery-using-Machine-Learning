
# 📁 Models Directory

This folder contains the pretrained model weights required to run the LULC Streamlit application.

## Included Models

- [`mobilenet.pth`](https://drive.google.com/uc?export=download&id=1x4Y5j2MMuT6dM7GmBO812FmtYZ3HhyFI) – Trained MobileNet backbone  
- [`efficientnet.pth`](https://drive.google.com/uc?export=download&id=1YqfINYYA0Q3k-zJ1gjhdOB3qzT8dTJDg) – Trained EfficientNet backbone  
- [`resnet50.pth`](https://drive.google.com/uc?export=download&id=1JSdv-vUldXOE3GjEJTlejm5XnSK0szfA) – Trained ResNet50 backbone  
- [`meta_clf.pkl`](https://drive.google.com/uc?export=download&id=1H5NAray6NclX1lrrDuvma9QaA70XQMkT) – Logistic Regression meta-classifier combining features of the three models  

> **Note:** Model files are **not included** in this repository due to size. Download the weights from the above links and place them in this folder.

## How to Use

1. Download the model weights and save them in this `models/` directory.  
2. Run the Streamlit application from the root directory:  

```bash
streamlit run app.py
```

3. The app will automatically load the models from this folder.

## Credits

- Dataset: **NWPU-RESISC45**  
- Training Platforms: **Kaggle** & **Google Colab**

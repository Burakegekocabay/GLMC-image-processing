# GLMC-image-processing
This project implements **texture analysis** using **Gray Level Co-occurrence Matrix (GLCM)**. 


It extracts texture features from images to enable **classification** of different surfaces, objects, or tissues.

# 🚀 Features

- Automatic dataset reading from subfolders (`dataset/<class_name>/`)
- Grayscale conversion of images
- GLCM computation at multiple orientations: 0°, 45°, 90°, 135°
- Extraction of 6+ texture features:
  - Contrast
  - Dissimilarity
  - Homogeneity
  - Energy
  - Correlation
  - Angular Second Moment (ASM)
- Extra feature: Standard deviation of grayscale intensity (`std`)
- Save extracted features into **ARFF** format compatible with Weka
- Fully modular and easy-to-extend Python code
  
# ⚡ Requirements
```bash
pip install numpy pandas opencv-python scikit-image
```

# How to Run
1️⃣ Run the main script to extract GLCM features and save as ARFF
```bash
python main.py
```
2️⃣ The ARFF file will be saved in the current directory as "glcm_features.arff". You can open it in Weka for classification and analysis


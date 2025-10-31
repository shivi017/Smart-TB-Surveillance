# Smart TB Surveillance: Integrating AI for Early Diagnosis and Spread Monitoring

This repository contains the Major Project for automated Tuberculosis detection from chest X-ray images.
## Project Role & Contributions
As the **Team Leader and Principal Developer**, I was responsible for the design, implementation, and evaluation of the deep learning framework used in this Major Project. My work involved integrating computational rigor with applied medical imaging to ensure reliable diagnostic performance.  
### Research & Design
Conducted an extensive study of CNN architectures and selected transfer learning models suitable for tuberculosis detection, including *ResNet50, DenseNet121, EfficientNetB0, InceptionV3, Xception,* and *MobileNetV2*.  

### Data Engineering
Developed preprocessing workflows for DICOM-to-PNG conversion, image normalization, and stratified dataset partitioning to maintain data balance and integrity.  

### Model Training & Analysis
Implemented K-Fold cross-validation and comparative performance evaluation across architectures, identifying *ResNet50* as the most robust baseline model.  

### Hyperparameter Optimization
Utilized **Keras Tuner (Hyperband)** for systematic tuning of learning rate, dropout, and dense layer configurations—resulting in an ROC-AUC improvement from **0.91 → 0.96** and an F1-score increase from **0.82 → 0.89**.  

### Final Model Development
Trained and validated the optimized *ResNet50* model achieving **96.5% accuracy**, and deployed it as a real-time **Streamlit web application** for end-user testing.  

### Leadership
Oversaw the technical workflow, coordinated the team’s research activities, and ensured documentation aligned with academic and reproducibility standards.

## Contents
- `di_convert.py` : DICOM to PNG conversion and preprocessing
- `data_split.py` : Dataset splitting (train/val/test)
- `model_comp.py` : Model training, K-Fold CV and comparison across architectures
- `hypertune_final.py` : Hyperparameter tuning using Keras Tuner (Hyperband)
- `final_training.py` : Final model training and evaluation on test set
- `app.py` : Streamlit app for deployment
- `requirements.txt` : Python dependencies
- `Final report for print.pdf` : Project report (included)
- `source code for major project.pdf` : Original source PDF (included)

## How to use
1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Review and adapt the `.py` scripts to match your dataset paths and model checkpoints.
3. Run data preprocessing and splitting:
   ```
   python di_convert.py
   python data_split.py
   ```
4. Train and evaluate models:
   ```
   python model_comp.py
   ```
5. Tune and train final model:
   ```
   python hypertune_final.py
   python final_training.py
   ```
6. Deploy the Streamlit app:
   ```
   streamlit run app.py
   ```

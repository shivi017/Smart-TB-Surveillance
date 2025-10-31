# 🫁 Smart TB Surveillance: Integrating AI for Early Diagnosis and Spread Monitoring

This repository contains the Major Project for automated Tuberculosis detection from chest X-ray images.

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

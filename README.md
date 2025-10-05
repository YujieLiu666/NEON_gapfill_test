

📘 **About this repository**

This is based on my previous paper published in _Agricultural and Forest Meteorology_. I have better organized the code and added a **Binder environment**, making it more user-friendly for everyone interested in gap-filling flux data using a machine learning model **XGBoost**. 

📬 **Questions or Collaborations?**

If you have any questions, suggestions, or are interested in collaborating, feel free to reach out! yujie.liu@nau.edu 

📝 **Citation**

_Liu, Yujie, et al. (2025). Robust filling of extra-long gaps in eddy covariance CO₂ flux measurements from a temperate deciduous forest using eXtreme Gradient Boosting. Agricultural and Forest Meteorology, 364, 110438._
https://doi.org/10.1016/j.agrformet.2025.110438 

🔗 What is Binder?

[Binder](https://mybinder.org/) is an open-source service that makes GitHub repositories interactive.
With just one click, users can launch a virtual compute environment with all dependencies installed. It is especially useful for teaching, code demonstrations, and sharing reproducible research. 

**Please click on the badge below**:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YujieLiu666/NEON_gapfill_test/HEAD?urlpath=lab&version=2)

- 🐍 **Python environment:** `environment.yml`

- 📂 **Input data:** `data_for_XGB_BART_NEON.csv`  
  - PPFD, Tair, and VPD are gapfilled using MDS  
  - NEE_for_gapfill is processed after IQR and u* filtering using REddyProc

- 📜 **Script:**  
  - All functions are stored in `function_XGB.py`  
  - Workflow: `workflow_XGB.ipynb` to run the functions

- 💾 **Output:**  
  - Gapfilled data: `FC_XGB_prediction.csv`  
  - Model after hyperparameter tuning: saved in subfolder `/XGB_models`  
  - FC_data_train_test: train and test data for 10-fold CV





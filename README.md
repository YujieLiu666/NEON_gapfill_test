

📘 **About this repository**

This is based on my previous paper published in _Agricultural and Forest Meteorology_. I have better organized the code in google colab, making it more user-friendly for everyone interested in gap-filling flux data using a machine learning model **XGBoost**. 
The work is supported by NEON ambassador program.

## 🚀 Get started using google colab

**Google Colab**  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YujieLiu666/NEON_gapfill_test/blob/main/workflow_XGB_google_colab.ipynb)


**Group discussion**
https://docs.google.com/document/d/1YJBQzLg3C42DVI0PhrJltVhKMfdSD9FJsxN3Uo1pBLc/edit?usp=sharing

**Feedback survey**
https://docs.google.com/forms/d/e/1FAIpQLSflPGHkiLr6P1Kc6ETOtzRVXk2m8Sp1zVbCUQdxT61F7kE98w/viewform

---




📬 **Questions or Collaborations?**

If you have any questions, suggestions, or are interested in collaborating, feel free to reach out! yujie.liu@nau.edu 

📝 **Citation**

_Liu, Yujie, et al. (2025). Robust filling of extra-long gaps in eddy covariance CO₂ flux measurements from a temperate deciduous forest using eXtreme Gradient Boosting. Agricultural and Forest Meteorology, 364, 110438._
https://doi.org/10.1016/j.agrformet.2025.110438 

- 🐍 **Python environment:** `environment.yml`

- 📂 **Input data:** `data_for_XGB_BART_NEON.csv`  
  - PPFD, Tair, and VPD are gapfilled using MDS  
  - NEE_for_gapfill is processed after IQR and u* filtering using REddyProc. You can find out how to do that from the tutorial one here: https://github.com/YujieLiu666/Bridginggap-flux
  - Processing input data using REddyProc? Tutorial can be found here!

- 📜 **Script:**  
  - All functions are stored in `function_XGB.py`  
  - Workflow: `workflow_XGB.ipynb` to run the functions

- 💾 **Output:**  
  - Model after hyperparameter tuning: saved in subfolder `/XGB_models`  

**Binder (experimental, in progress)**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YujieLiu666/NEON_gapfill_test/HEAD?urlpath=lab&version=2)








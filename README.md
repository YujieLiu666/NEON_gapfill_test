[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YujieLiu666/NEON_gapfill_test/HEAD)



citation: https://doi.org/10.1016/j.agrformet.2025.110438 
python environment: environment.yml

Input data: data_for_XGB_BART_NEON.csv
- PPFD, Tair and VPD are gapfilled using MDS;
- NEE_for_gapfill is processed after IQR and u* filtering using REddyProc.

Script:
- All the functions are stored in function_XGB.py.
- workflow: workflow_XGB.ipynb to run the functions.


Output:
- gapfilled data: FC_XGB_prediction.csv
- variable importance: FC_feature_importances.csv
- model after hyperparameter tuning: /XGB_models
- FC_data_train_test: train and test data for 10-fold CV
- learning curve folder: learning curve for each fold


To be updated: 
- R_XGB.Rmd (experimental): converting python codes to R codes.
- current issue: bad hyperparameter tuning 

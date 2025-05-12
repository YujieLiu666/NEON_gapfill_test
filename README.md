[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YujieLiu666/NEON_gapfill_test/HEAD)

citation: https://doi.org/10.1016/j.agrformet.2025.110438 

input data: data_for_XGB_BART_NEON.csv
- PPFD, Tair and VPD are gapfilled using MDS;
- NEE_for_gapfill is processed after IQR and u* filtering using REddyProc.
- All the functions are stored in function_XGB.py.
- workflow: workflow_XGB.ipynb to run the functions.
- R_XGB.Rmd (experimental): converting python codes to R codes.

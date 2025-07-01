## Description

This is the implementation of the OMIDIENT paper: **"OMIDIENT: multiOMics Integration for cancer by
DIrichlet auto-ENcoder neTworks"**

See code folder for the codes implemented for this paper. 
The folder code_two_omics is used for when the data modality is two.

For any question contact **negar7918@gmail.com**

Here we show how to run the different implemented method for the breast cancer data. 
For reproducing the other data experiments contact the above email.

## How to run each method

-----MOCSS (AE)-----
1. Run code/ae/hyperparam_tuning_mocss.py
2. Run code/ae/mocss_original_refactored.py
3. Run code/ae/mocss_missing_data.py

-----LassoAE-----
1. Activate Lasso by setting "lasso = True" in the beginning of the below files
2. Run code/ae/hyperparam_tuning_mocss.py
2. Run code/ae/mocss_original_refactored.py

-----GamDirVae-----
1. Run code/gamma_dirvae/hyperparam_tuning_gamma_dirvae_cancer.py
2. Run code/gamma_dirvae/gamma_dirvae_cancer.py

-----LapDirVae-----
1. Run code/Laplace_dirvae/hyperparam_tuning_lap_dirvae_cancer.py
2. Run code/Laplace_dirvae/lap_dirvae_cancer.py

-----VAE-----
1. Run code/vae/hyperparam_tuning_mocss_vae.py
2. Run code/vae/mocss_vae.py

-----ProdGamDirVae (OMIDIENT)-----
1. Run code/prod_gamma_dirvae/hyperparam_tuning_prod_gamma_dirvae_cancer.py
2. Run code/prod_gamma_dirvae/prod_gamma_dirvae_cancer.py
3. Run code/prod_gamma_dirvae/prod_gamma_dirvae_cancer_missing_data.py

## Python Requirements

      - lazy-loader==0.4
      - llvmlite==0.42.0
      - markupsafe==2.1.5
      - matplotlib==3.8.2
      - multidict==6.0.5
      - munkres==1.1.4
      - networkx==3.2.1
      - numba==0.59.1
      - numpy==1.26.3
      - packaging==24.0
      - pandas==2.2.0rc0
      - pillow==10.3.0
      - pyparsing==3.1.2
      - scikit-image==0.24.0
      - scikit-learn==1.4.0rc1
      - scipy==1.12.0
      - seaborn==0.13.2
      - shap==0.46.0
      - slicer==0.0.8
      - sympy==1.13.1
      - torch==2.5.0
      - torch-geometric==2.5.2
      - torchvision==0.20.0
      - tqdm==4.66.2
      - tzdata==2024.1
      - umap-learn==0.5.5
#Dataset for GNN: https://drive.google.com/drive/folders/11KJCix7QN3zJaKRrv1Hx14P5OVdB3cty?usp=sharing


# GIS
Dataset for MATLAB code: Data will be made available on request.

Dust Emission Forecasting Model

1. Overview
This repository contains a MATLAB-based machine-learning pipeline for forecasting dust emissions using geospatial and environmental data (2000–2021). The model integrates multiple classifiers (Random Forest, SVM, KNN, multinomial Naive Bayes) and performs feature importance analysis to identify key drivers of dust events.

3. Repository Structure
dust-emission-forecasting/  
│── data/                   # Input datasets (not included due to size)  
│   ├── polygon1kmfinal.shp  # Region of Interest (ROI) shapefile  
│   ├── aod2000-2021.tif     # Dust observation data (GeoTIFF)  
│   └── *.mat                # Environmental variables (NDVI, rainfall, etc.)  
│  
│── src/  
│   ├── forecast-dust-emission.m  # Main MATLAB script  
│   └── utils/                   # Helper functions (SMOTE, performance metrics)  
│  
│── results/                  # Output figures & model performance  
│   ├── ROC_curves/           # Receiver Operating Characteristic plots  
│   └── feature_importance/   # Rankings of environmental predictors  
│  
│── LICENSE.md                # MIT License  
│── CITATION.cff              # Citation metadata for academic use  
│── requirements.txt          # MATLAB toolbox dependencies  
└── README.md                 # This file


2. Repository Structure
dust-emission-forecasting/  
│── data/                   # Input datasets (not included due to size)  
│   ├── polygon1kmfinal.shp  # Region of Interest (ROI) shapefile  
│   ├── aod2000-2021.tif     # Dust observation data (GeoTIFF)  
│   └── *.mat                # Environmental variables (NDVI, rainfall, etc.)  
│  
│── src/  
│   ├── forecast-dust-emission.m  # Main MATLAB script  
│   └── utils/                   # Helper functions (SMOTE, performance metrics)  
│  
│── results/                  # Output figures & model performance  
│   ├── ROC_curves/           # Receiver Operating Characteristic plots  
│   └── feature_importance/   # Rankings of environmental predictors  
│    
│── CITATION.cff              # Citation metadata for academic use  
│── requirements.txt          # MATLAB toolbox dependencies  
└── README.md                 # This file

3. Requirements
Software
MATLAB R2021a+ (tested on R2023a)

Required Toolboxes:

Statistics and Machine Learning Toolbox

Parallel Computing Toolbox (optional, for speedup)

Mapping Toolbox (for geospatial operations)

Hardware
Minimum: 8 GB RAM, 4-core CPU

Recommended: 16+ GB RAM for large datasets

4. Installation & Execution
Step 1: Clone the Repository
git clone https://github.com/agmlcenter/GIS/tree/main/Dust_Source_Emission_Forecasting.git
cd Dust_Source_Emission_Forecasting
Step 2: Load Data
Place the following files in /data/:

polygon1kmfinal.shp (ROI boundary)

aod2000-2021.tif (dust observations)

.mat files (e.g., NDVI_2000_2021.mat, Rainfall_2000_2021.mat)

Step 3: Run the Model
Open MATLAB and navigate to the repository.

Execute the main script:
run('src/forecast-dust-emission.m')  

5. Methodology
Data Preprocessing
Spatial Resampling: Uniform 1km resolution (Pixel_Resize=1)

Temporal Aggregation: Monthly dust events (2000–2021)

Class Balancing: SMOTE oversampling for imbalanced dust/no-dust labels

Machine Learning Models:
Random Forest	
SVM	
KNN	
Multinomial Naive Bayes	
Validation
Metrics: AUC-ROC, accuracy, precision, recall

Time-based Split:
Training: 2000–2020
Testing: 2021 (holdout set)
Uncertainty Quantification: Dempster-Shafer fusion of classifier outputs

Version Information
Current Version: v1.0 (tagged as submission-version)

MATLAB Compatibility: R2021a–R2023a


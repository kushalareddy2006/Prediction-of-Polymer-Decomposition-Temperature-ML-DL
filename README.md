# Prediction-of-Polymer-Decomposition-Temperature-ML-DL

Prediction of Polymer Decomposition Temperature Using Machine Learning and Deep Learning

Overview:
This project aims to predict the decomposition temperature (Td) of polymers, which indicates their thermal stability, using machine learning (ML) and deep learning (DL) models. The dataset contains 10,050 polymer samples, each described by SMILES-based molecular structures, composition, and various thermal, mechanical, and gas permeability features. The goal is to create a reliable predictive framework that can accelerate polymer research and reduce the need for expensive experimental testing.

Methodology:
The dataset was obtained from the Polymer Genome Database and includes 18–23 features per sample. Data preprocessing steps involved validating SMILES strings using RDKit, handling missing values through mean imputation, removing duplicates, and scaling numerical features. Molecular descriptors such as molecular weight and topological polar surface area were extracted from SMILES strings.

Several ML and DL models were implemented and compared:

Machine Learning Models: XGBoost, Random Forest, and K-Nearest Neighbors (KNN)

Deep Learning Models: Multilayer Perceptron (MLP), Deep Neural Network (DNN), and Recurrent Neural Network (RNN)

Hybrid Architectures: CNN, LSTM, GRU, CNN-LSTM, CNN-GRU, and CNN-Dense

Each model was evaluated using metrics like Root Mean Squared Error (RMSE) and Coefficient of Determination (R²). Visual tools such as confusion matrices, learning curves, and scatter plots were used to assess model performance.

Results:

XGBoost achieved the best performance with an R² of 0.960, showing high accuracy for tabular polymer data.

The MLP model performed well among deep learning approaches with an R² of 0.875.

LSTM and CNN-Dense models showed moderate performance, with R² values around 0.76–0.80, mainly due to the non-sequential nature of the dataset.
A command-line interface (CLI) was also developed to allow real-time Td predictions for new SMILES inputs.

Technologies Used:
Python 3.7+, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, RDKit, Matplotlib, Seaborn, and Joblib.
The project was developed and tested using Jupyter Notebook and Google Colab.
Models were saved in .pkl format for ML and .h5 format for DL models.

Impact:
This project provides a fast and cost-effective alternative to experimental testing for determining polymer decomposition temperature. It helps researchers quickly screen and design new polymer materials with improved thermal stability, supporting sustainable material development in industries like aerospace, electronics, and automotive.

Future Work:

Integrate Graph Neural Networks (GNNs) for better molecular representation.

Expand the dataset to enhance model generalization.

Improve model interpretability using SHAP or feature importance analysis.

Develop an interactive web application for easy prediction of polymer properties.

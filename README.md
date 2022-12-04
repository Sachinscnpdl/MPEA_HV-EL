# MPEA_HV-EL
# Notebook files (.ipynb)
# 1. "common_functions.ipynb": notebook file consisting all the common codes from elongation and hardness training, prediction and probablistic model
# 2. "Hardness_Train_model.ipynb" and "Elongation_Train_Model.ipynb": codes for training hardness and elongation models. Calls function from #1.
# 3. "Hardness_Prediction_RMPEA.ipynb" and "Elongation_Prediction_RMPEA.ipynb": Hardness and Elongation prediction of Refractory Multi-Principal Element Alloys (RMPEA) using #2.
# 4. "Hardness_Probablistic_Model.ipynb" and "Elongation_Probablistic_Model.ipynb": Bayesian neural network (BNN) for probablistic hardness and elongation models.

# CSV files (.csv)
# 1. "hardness.csv" and "elongation.csv": Train + Test data collected for hardness and elongation model.
# 2. "Miedema.csv" and "input_parameters.csv": Properties of elements used in feature calculations.
# 3. "test_new_hv.csv": Evaluation data of hardness and elongation model of new alloy composition apart from Train+Test.
# 4. "rhea_plot.csv": HEA/MPEA composition for hardness and elongation prediction on ternary plots.
# 5. "pred.csv": Alloy compositions inputs for predictions used molecular dynamics analysis.

# Folder files
# 1. "elongation_model_files": consists of .h5 files and pickle files of ensemble elongation models and standarization and feature selections.
# 2. "hardness_model_files": consists of .h5 files and pickle files of ensemble hardness models and standarization and feature selections.
# 3. "plots" and "triangular plots": consists of plots and graphs of both elongation and hardness design.

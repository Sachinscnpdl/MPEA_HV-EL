import pkg_resources
import subprocess
import sys

from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import pymatgen
import matminer
import tensorflow as tf

pd.set_option('display.max_columns', None)
import numpy as np
import os
import matplotlib.pyplot as plt

# ef= ElementFraction()
# stc = StrToComposition()

# Add the function.py file
from app_functions import *
# from prediction_ML import *
#############################################################################
# Header for the Website
#st.header(':blue[Optimizing Synergy Between Hardness and Ductility in MPEAs] ')
#st.header('Toolkit for Exploratory Design and Discovery of Piezoelectric Materials ')
# Using HTML formatting to add color to the font in st.header
st.markdown('<h1 style="color:purple;">Mechanical Properties Prediction of MPEAs </h1>', unsafe_allow_html=True)
# Add a dropdown to select a pre-defined formula

import streamlit as st
import pandas as pd

# Create two tabs in the sidebar
tab_options = ["New alloy design", "Synergy Optimization"]
selected_tab = st.sidebar.radio("Select Tab", tab_options)

# Initialize empty DataFrame to store selected formulas
df_selected_formulas = pd.DataFrame()

# Check the selected tab
if selected_tab == "New alloy design":
    # Add a dropdown to select a pre-defined formula in the sidebar
    predefined_formulas = ['CoCrNi', 'CoCrNiNb0.2', 'CoCrNiNb0.3', 'CoCrNiNb0.7']
    selected_predefined_formula = st.sidebar.selectbox('Select a pre-defined formula', predefined_formulas)

    fabrication_type_options = ["CAST", "POWDER", "ANNEAL", "WROUGHT", "OTHER"]
    selected_fabrication_type = st.sidebar.selectbox('Select Fabrication Type:', fabrication_type_options)

    # If a pre-defined formula is selected, add it to the DataFrame
    if selected_predefined_formula:
        # Check if the DataFrame is empty
        if df_selected_formulas.empty:
            df_selected_formulas = pd.DataFrame({'S.N': [1], 'Alloys': [selected_predefined_formula], 'Fabrication_type': [None]})
        else:
            df_selected_formulas.at[len(df_selected_formulas)-1, 'Alloys'] = selected_predefined_formula

    # Update the Fabrication_Type column of the last row with the selected Fabrication_type
    if selected_fabrication_type:
        df_selected_formulas.at[len(df_selected_formulas)-1, 'Fabrication_type'] = selected_fabrication_type

################################################################################################
    
    df_mpea = df_selected_formulas
    df_mpea = featurization(df_mpea)
    
    df_mpea = df_element_number(df_mpea)
    df_mpea = data_elimination(df_mpea)
    df_mpea = fab_cluster(df_mpea)
    df_mpea, df_input_target = properties_calculation(df_mpea)

    hardness = prediction_model_new(df_mpea, predict='hardness')
    elongation = prediction_model_new(df_mpea, predict='elongation')
    hardness = round(hardness[0],2)
    elongation = round(elongation[0],2)
    
    # """ 
    # # Prediction Results!
    # """
    # Define the text style for hardness and elongation with values and units

    
    # Define the text style for hardness and elongation with values and units
    hardness_style = "<h2 style='color:green; font-size:24px;'>{} HV</h2>".format(hardness)
    elongation_style = "<h2 style='color:green; font-size:24px;'>{} %</h2>".format(elongation)
    
    # Display the styled text using st.markdown on the same line
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>The hardness is:</h2> {}".format(hardness_style), unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>The elongation is:</h2> {}".format(elongation_style), unsafe_allow_html=True)



    
    # Display other content as needed
    st.markdown("<h3 style='color:red;'>Composition-based feature vector</h3>", unsafe_allow_html=True)
    st.write(df_input_target)


    # Define the property names in LaTeX format
    property_names_latex = [
        r"\text{Mean Atomic Radius difference } (\delta)",
        r"\text{Electronegativity asymmetry } (\Delta \chi)",
        r"\text{Melting Temperature asymmetry } (\Delta T_m)",
        r"\text{Average Melting Temperature } (T_{m}(K))",
        r"\text{Valence Electron Concentration } (VEC)",
        r"\text{Average Atomic Number } (AN)",
        r"\text{Thermal Conductivity } (K)",
        r"\text{Average Bulk Modulus } (B)",
        r"\text{Bulk Modulus Asymmetry } (\Delta B)",
        r"\text{Average Shear Modulus } (G)",
        r"\text{Shear Modulus Asymmetry } (\Delta G)",
        r"\text{Entropy of Mixing asymmetry } (\Delta S_{mix})",
        r"\text{Geometrical Parameter } (\lambda)",
        r"\text{Enthalpy of Mixing asymmetry } (\Delta H_{mix})",
        r"\text{Dimensionless parameter } (\Omega)",
    ]
    
    # Create a list of values (replace this with your actual values)
    values = df_input_target.iloc[0, 1:].tolist()
    
    # Display the property names with left alignment and values rounded to 4 decimal places
    for i in range(len(property_names_latex)):
        formatted_value = "{:.3}".format(values[i])  # Format value to 4 decimal places
        st.latex(r"\quad {} : {}".format(property_names_latex[i], formatted_value))




       
#####################################################################

     
###############################################

if selected_tab == "Synergy Optimization":
    """
    
    # Welcome to PiezoTensorNet - Piezoelectric performance finetuning!
    """
    
    base_material_options = ["BaTiO3", "AlN"]
    base_composition = st.sidebar.selectbox("Base Piezo-material", base_material_options)

    first_dopants_options = ["Mo", "Mg", "Ti", "Zr", "Hg"]
    first_dopant = st.sidebar.selectbox("First Dopants", first_dopants_options)

    second_dopants_options = ["Mo", "Mg", "Ti", "Zr", "Hg"]
    second_dopant = st.sidebar.selectbox("Second Dopants", second_dopants_options)
    
    # Perform actions or display content based on the selected options
    st.write("Selected Base Piezo-material:", base_composition)
    st.write("Selected First Dopant:", first_dopant)
    st.write("Selected Second Dopant:", second_dopant)
    # Additional code for this tab

    if second_dopant:
        # Both element 1 and element 2 are supplied
        cat = 'B'
        point = 'hextetramm'
        order = [2, 0]
        cat, sub, tensor_eo = two_dopants_ternary(base_composition, first_dopant, second_dopant, cat, point, order)
        st.write("Results for two dopants:")
        st.write("Category:", cat)
        st.write("Subcategory:", sub)
        st.write("Tensor EO:", tensor_eo)
    else:
        # Only element 1 is supplied
        cat = 'B'
        point = 'hextetramm'
        order = [2, 2]
        tensor_eo, target_1, target_33_1, target_31_1 = single_dopants_new(base_composition, first_dopant, cat, point, order)
        st.write("Results for single dopant:")
        st.write("Tensor EO:", tensor_eo)
        st.write("Target 1:", target_1)
        st.write("Target 33_1:", target_33_1)
        st.write("Target 31_1:", target_31_1)

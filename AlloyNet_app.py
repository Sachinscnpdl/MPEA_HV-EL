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
st.markdown('<h1 style="color:purple;">Optimizing Synergy Between Hardness and Ductility in MPEAs </h1>', unsafe_allow_html=True)
# Add a dropdown to select a pre-defined formula

# Create two tabs in the sidebar
tab_options = ["New alloy design", "Synergy Optimization"]
selected_tab = st.sidebar.radio("Select Tab", tab_options)

# Initialize empty DataFrame to store selected formulas
df_selected_formulas = pd.DataFrame()

# Add input block for "New alloy design"
if selected_tab == "New alloy design":
    # Add a dropdown to select a pre-defined formula
    predefined_formulas = ['CoCrNi','CoCrNiNb0.2', 'CoCrNiNb0.3', 'CoCrNiNb0.7']
    selected_predefined_formula = st.selectbox('Select a pre-defined formula', predefined_formulas)

    # If a pre-defined formula is selected, add it to the DataFrame
    if selected_predefined_formula:
        df_selected_formulas = pd.concat([df_selected_formulas, pd.DataFrame({'S.N': [len(df_selected_formulas) + 1], 'formula_pretty': [selected_predefined_formula]})], ignore_index=True)
################################################################
    
    # Display the selected formulas
    if not df_selected_formulas.empty:
        st.write('Selected Formulas:')
        st.dataframe(df_selected_formulas)
    
    df_mpea = df_selected_formulas
    df_mpea = featurization(df_mpea)
    
    df_mpea = df_element_number(df_mpea)
    df_mpea = data_elimination(df_mpea)
    df_mpea = fab_cluster(df_mpea)
    df_mpea, df_input_target = properties_calculation(df_mpea)
    print("Properties Calculated")
    hardness = prediction_model_new(df_mpea, predict='hardness')





    
    """
    
    # Welcome to PiezoTensorNet!
    
    """
    
    st.write("Crystal Structure is :", subcategories[0][0])
    #####################################################################
    # In[15]:
    print("Tensor Predictionsin Progress")
    
    my_tensor = np.array(y_tensor[0])
    
    ####################################################################
    """
    ## The Piezo Tensor is
    
    """
    
    # Display LaTeX matrix using st.latex()
    st.latex(latex_tensor_visual(my_tensor))
    #####################################################################
    
    # Piezo Tensor Representations
    if subcategories[0][0]=='cubic':
        image = "plots/CAT_A.png"
    elif subcategories[0][0]=='tetra42m':
        image = "plots/CAT_A.png"    
    elif subcategories[0][0]=='ortho222':
        image = "plots/CAT_A.png"
        
    elif subcategories[0][0]=='orthomm2':
        image = "plots/orthomm2.png"
    elif subcategories[0][0]=='hextetramm':
        image = "plots/hextetra.png"
    
    caption = "Piezoelectric Tensor Visualization"
    size = (300, 200)  # Custom size in pixels
    position = "right"  # Options: "left", "centered", "right"
    
    st.image(image,width=200, caption=caption)
    
    # st.image(image, caption="Image Caption", width=10, use_column_width=True)
    
    #####################################################################
    
    
    ######################################################################################################
    
    import subprocess
    
    # Define the path to the file
    file_path = "~/Sachin Research/Piezoelectric_Research/Piezoelectric_codes/Codes_May 1/Piezoelectric Tensors/"
    
    # Create a NumPy array
    # my_array = np.array([1, 2, 3, 4, 5])
    
    # Save the NumPy array as a tensor file
    tensor_path = 'my_tensor.npy'
    np.save(tensor_path, my_tensor)
    
    # Add a download button to the Streamlit app
    if st.button("Download NumPy Array as Tensor"):
        with open(tensor_path, "rb") as file:
            contents = file.read()
            st.download_button(label="Click here to download", data=contents, file_name="my_array.npy")
            
    #####################################################################################################
    # Crystal Rotations
    from crystal_rotation import *
    
    # Take an input from the user
    crystal_rotations = st.sidebar.checkbox("Perform Crystal rotation for rotated tensor")
    if crystal_rotations:
       # Create input fields in a compact and aligned layout
        cols = st.sidebar.columns([1, 1, 1, 1])
        cols[0].write('<p style="margin-bottom: -0.2em;">Euler Angle:</p>', unsafe_allow_html=True)  
        cols[1].write("Psi ")
        cols[2].write("Theta")
        cols[3].write("Phi")
        psi = cols[1].text_input("", value="30")
        theta = cols[2].text_input("", value="90")
        phi = cols[3].text_input("", value="150")
    
        # Perform tensor rotation with crystal rotations
        tensor_prime = tensor_rotation(my_tensor, psi=float(psi), theta=float(theta), phi=float(phi))
    
    #     tensor_rotation_plot(my_tensor, phi_vals = 30,order=[1,1])
    
        # Display LaTeX matrix using st.latex()
        st.latex(latex_tensor_visual(tensor_prime))
        
        phi = st.sidebar.slider("Enter angle phi:", min_value=0.0, max_value=360.0, value=0.0, step=10.0)
    
        block = st.sidebar.columns([1, 1, 1])
        block[0].write('<p style="margin-bottom: -0.2em;">Show plot of Rotated tensor:</p>', unsafe_allow_html=True)
        block[1].write("Row element", unsafe_allow_html=True)
        block[2].write("Column element", unsafe_allow_html=True)
    
        i = block[1].text_input("Row", value="1", key="row_element")
        j = block[2].text_input("Column", value="1", key="column_element")
    
        i = int(i)
        j = int(j)
    
        crystal_plot, max_e, max_theta, max_psi, phi_vals = tensor_rotation_plot(my_tensor, phi = phi, order=[i-1,j-1])
        st.plotly_chart(crystal_plot)
        
        rot_optimization = tensor_rotation_optimization(my_tensor, order=[i-1,j-1])
        st.pyplot(rot_optimization)
    

####################################################
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

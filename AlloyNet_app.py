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

st.set_page_config(layout="wide")
# Header for the Website
#st.header(':blue[Optimizing Synergy Between Hardness and Ductility in MPEAs] ')
#st.header('Toolkit for Exploratory Design and Discovery of Piezoelectric Materials ')
# Using HTML formatting to add color to the font in st.header
st.image("plots/logo.png",width=150)
st.markdown('<h1 style="color:purple;">AlloyManufacturingNet </h1>', unsafe_allow_html=True)
# Add a dropdown to select a pre-defined formula

# Create two tabs in the sidebar
tab_options = ["New alloy design", "HV-EL Synergy Optimization"]
selected_tab = st.sidebar.radio("Select Tab", tab_options)

# Initialize empty DataFrame to store selected formulas
df_selected_formulas = pd.DataFrame()
st.session_state.clear()
# Check the selected tab
if selected_tab == "New alloy design":
    # Add a dropdown to select a pre-defined formula in the sidebar
    predefined_formulas = ['CoCrNi', 'CoCrNiNb0.2', 'CoCrNiNb0.3', 'CoCrNiNb0.6', 'CoCrNiNb0.7']
    # selected_predefined_formula = st.sidebar.selectbox('Select a pre-defined formula', predefined_formulas)



####################################
    # Initialize DataFrame to store selected formulas if not already initialized
    if 'df_selected_formulas' not in st.session_state:
        st.session_state.df_selected_formulas = pd.DataFrame(columns=['S.N', 'Alloys', 'Fabrication_type'])
    
    # List of elements to be included in the dropdowns
    elements_list = ["Sc", "Pd", "Re", "Ca", "Y", "Li", "Zn", "Mg", "C", "Sn", "W", "Si", "Hf", "Ta", 
                     "Mn", "Cu", "V", "Zr", "Mo", "Nb", "Co", "Al", "Ti", "Cr", "Fe", "Ni"]
    
    # Add a dropdown for the user to choose between predefined formula and new alloy in the sidebar
    option = st.sidebar.selectbox('Choose an option:', ['Select a predefined formula', 'Define new alloy'])
    
    if option == 'Select a predefined formula':
        # Add a dropdown to select a pre-defined formula in the sidebar
        predefined_formulas = ['CoCrNi', 'CoCrNiNb0.2', 'CoCrNiNb0.3', 'CoCrNiNb0.6', 'CoCrNiNb0.7']
        selected_predefined_formula = st.sidebar.selectbox('Select a pre-defined formula', predefined_formulas)
        
        # If a pre-defined formula is selected, add it to the DataFrame
        if selected_predefined_formula:
            new_row = {'S.N': len(st.session_state.df_selected_formulas) + 1, 'Alloys': selected_predefined_formula, 'Fabrication_type': None}
            # st.session_state.df_selected_formulas = st.session_state.df_selected_formulas.append(new_row, ignore_index=True)
            
            # st.write('Selected predefined formula:', selected_predefined_formula)
    else:
        # Add inputs to define a new alloy within the sidebar
        st.sidebar.write('Define a new alloy')
        
        elements = []
        compositions = []
        
        num_elements = st.sidebar.number_input('Number of elements', min_value=3, max_value=10, step=1, value=3)
        
        default_elements = ['Co', 'Cr', 'Ni']
        default_compositions = ['1', '1', '1']
        
        for i in range(num_elements):
            col1, col2 = st.sidebar.columns(2)
            if i < 3:
                element = col1.selectbox(f'Element {i+1}', elements_list, index=elements_list.index(default_elements[i]), key=f'element_{i+1}')
                composition = col2.text_input(f'Composition {i+1}', default_compositions[i], key=f'composition_{i+1}')
            else:
                element = col1.selectbox(f'Element {i+1}', elements_list, key=f'element_{i+1}')
                composition = col2.text_input(f'Composition {i+1}', '1', key=f'composition_{i+1}')
            elements.append(element)
            compositions.append(composition)
        
        # Concatenate elements and compositions into a single string
        alloy_string = ''.join([f'{el}{comp}' for el, comp in zip(elements, compositions)])
        
        # If a new alloy is defined, add it to the DataFrame
        if alloy_string:
            new_row = {'S.N': len(st.session_state.df_selected_formulas) + 1, 'Alloys': alloy_string, 'Fabrication_type': None}
    st.session_state.df_selected_formulas = st.session_state.df_selected_formulas.append(new_row, ignore_index=True)
    
    st.write('Defined new alloy:', alloy_string)
    
    # Display the DataFrame only if it has necessary columns
    # if 'Alloys' in st.session_state.df_selected_formulas.columns and 'Fabrication_type' in st.session_state.df_selected_formulas.columns:
        # st.write(st.session_state.df_selected_formulas)
    
    # Additional sidebar options
    fabrication_type_options = ["CAST", "POWDER", "ANNEAL", "WROUGHT", "OTHER"]
    selected_fabrication_type = st.sidebar.selectbox('Select Fabrication Type:', fabrication_type_options)
    
    elongation_test_options = ["Default", "Tensile", "Compression"]
    el_test = st.sidebar.selectbox('Test for Ductility:', elongation_test_options)
    
    # Update the Fabrication_Type column of the last row with the selected Fabrication_type
    if 'Fabrication_type' in st.session_state.df_selected_formulas.columns:
        if selected_fabrication_type and not st.session_state.df_selected_formulas.empty:
            st.session_state.df_selected_formulas.at[len(st.session_state.df_selected_formulas)-1, 'Fabrication_type'] = selected_fabrication_type
    
    # Display updated DataFrame
    if 'Alloys' in st.session_state.df_selected_formulas.columns and 'Fabrication_type' in st.session_state.df_selected_formulas.columns:
        st.write(st.session_state.df_selected_formulas)
################################################################################################
    from io import StringIO
    df_mpea = df_selected_formulas
    df_mpea = featurization(df_mpea)
    
    df_mpea = df_element_number(df_mpea)
    df_mpea = data_elimination(df_mpea)
    df_mpea = fab_cluster(df_mpea)
    df_mpea, df_input_target = properties_calculation(df_mpea)

    hardness = prediction_model_new(df_mpea, predict='hardness', el_test=el_test)
    elongation = prediction_model_new(df_mpea, predict='elongation', el_test=el_test)
    hardness = round(hardness[0],2)
    elongation = round(elongation[0],2)

    
    # """ 
    # # Prediction Results!
    # """
    # Define the text style for hardness and elongation with values and units

    alloy_style = "<h2 style='color:green; font-size:24px;'>{} </h2>".format(selected_predefined_formula)
    manufacturing_style = "<h2 style='color:green; font-size:24px;'>{} </h2>".format(selected_fabrication_type)
    # Display the styled text using st.markdown on the same line
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Multi-principal Element Alloy:</h2> {}".format(alloy_style), unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Fabrication Type:</h2> {}".format(manufacturing_style), unsafe_allow_html=True)

    # Define the text style for hardness and elongation with values and units
    hardness_style = "<h2 style='color:green; font-size:24px;'>{} HV</h2>".format(hardness)
    elongation_style = "<h2 style='color:green; font-size:24px;'>{} %</h2>".format(elongation)
    
    # Display the styled text using st.markdown on the same line
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Hardness:</h2> {}".format(hardness_style), unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Ductility:</h2> {}".format(elongation_style), unsafe_allow_html=True)

    
    # Display other content as needed
    st.markdown("<h3 style='color:red;'>Composition-based feature vector:</h3>", unsafe_allow_html=True)
    # st.write(df_input_target)

    property_names_latex = [
        r"Mean Atomic Radius difference (δ) ",
        r"Electronegativity asymmetry (Δ χ) ",
        r"Melting Temperature asymmetry (Δ T)  ",
        r"Average Melting Temperature  (T, K) ",
        r"Valence Electron Concentration  (VEC) ",
        r"Average Atomic Number  (AN) ",
        r"Thermal Conductivity  (K) ",
        r"Average Bulk Modulus  (B) ",
        r"Bulk Modulus Asymmetry  (Δ B) ",
        r"Average Shear Modulus  (G) ",
        r"Shear Modulus Asymmetry  (Δ G) ",
        r"Entropy of Mixing asymmetry (Δ S) ",
        r"Geometrical Parameter  (λ) ",
        r"Enthalpy of Mixing asymmetry  (Δ H) ",
        r"Dimensionless parameter  (Ω) ",
    ]
    
    # Create a list of values (replace this with your actual values)
    values = df_input_target.iloc[0, 1:16].tolist()
    
    # Ensure that property_names_latex and values have the same length
    if len(property_names_latex) != len(values):
        st.error("The number of property names and values should be the same.")
    else:
        # Create a DataFrame with LaTeX-formatted property names and values
        df = pd.DataFrame({"Feature Names": property_names_latex, "Value": values})
    
        # Add an index starting from 1
        df.index = range(1, len(df) + 1)
    
        # Define custom CSS styles for the table cells (no vertical lines, top and bottom horizontal rules)
        cell_style = (
            "<style>"
            "table.dataframe {border-collapse: collapse; width: 100%;}"
            "table.dataframe th, table.dataframe td {border: none; border-bottom: 1px solid #ddd; text-align: left; padding: 8px;}"
            "table.dataframe th {background-color: #f2f2f1;}"
            "table.dataframe tr:nth-child(even) {background-color: #f2f2f2;}"
            "</style>"
        )
        st.markdown(cell_style, unsafe_allow_html=True)
    
        # Display the DataFrame as a beautiful table
        st.dataframe(df, height=570)  # You can adjust the height as needed
        st.write("Pugh's Ratio:", round(df_input_target.iloc[0, 8] / df_input_target.iloc[0, 7],2))
       
##################################################################### ##################################################################### 
###############################################

if selected_tab == "HV-EL Synergy Optimization":

    base_material_options = ["(VNbTa)", "(ZrHfNb)", "(MoNbTa)", "(CrFeCoNi)", "(CoCrNi)"]
    base_composition = st.sidebar.selectbox("Base MPEA", base_material_options)

    fabrication_type_options = ["CAST", "POWDER", "ANNEAL", "WROUGHT", "OTHER"]
    fab_type = st.sidebar.selectbox('Select Fabrication Type:', fabrication_type_options)

    first_dopants_options = ["Zr", "Mo", "W", "Ta",  "Hg", "Mo", "Ti"]
    first_dopant = st.sidebar.selectbox("First Dopants", first_dopants_options)

    second_dopants_options = ["W", "Ta", "Mg", "Ti", "Zr", "Hg"]
    second_dopant = st.sidebar.selectbox("Second Dopants", second_dopants_options)

    elongation_test_synergy = ["Default", "Tensile", "Compression"]
    el_test_synergy = st.sidebar.selectbox('Test for Ductility:', elongation_test_synergy)

    colorset_1 = ['Picnic', 'Viridis', 'Rainbow', 'Blackbody', 'Jet', 'Portland', 'Cividis', 'Electric']
    hv_colorset = st.sidebar.selectbox('Select Colorset for HV:', colorset_1)

    colorset_2 = ['Jet', 'Picnic', 'Viridis', 'Rainbow', 'Blackbody',  'Portland', 'Cividis', 'Electric']
    el_colorset = st.sidebar.selectbox('Select Colorset for EL:', colorset_2)
    
    pole_labels=[ base_composition,first_dopant,second_dopant]
     
    # st.dataframe(df_test, height=570)
    ternary_hv, dopant_input,df_alloy, dopant_pred_hv = ternary_plot(fab_cat=fab_type, pole_labels=[ base_composition,first_dopant,second_dopant],model_of='hardness', colorscale=hv_colorset,el_test=el_test_synergy)
    ternary_el, dopant_input,df_alloy, dopant_pred_el = ternary_plot(fab_cat=fab_type, pole_labels=[ base_composition,first_dopant,second_dopant],model_of='elongation', colorscale=el_colorset,el_test=el_test_synergy)

    dopant_input['Alloys'] = df_alloy
    dopant_input['Hardness'] = dopant_pred_hv
    dopant_input['Elongation'] = dopant_pred_el
    # st.write(dopant_pred)
    
    st.plotly_chart(ternary_hv)
    st.plotly_chart(ternary_el)
    ################# Add design chart ############################
   
    st.image("plots/synergy.png",width=600)
    #######################################################
    import numpy as np
    import plotly.express as px
    
    # Sample data (replace with your actual data)

    # Create a Scatter Plot using Plotly Express
    fig = px.scatter(x=dopant_pred_el, y=dopant_pred_hv, labels={'x':'Elongation', 'y':'Hardness'})
    
    # Create custom hover text that includes the index value
    hover_text = [f"Index: {i}<br>Elongation: {x}<br>Hardness: {y}" for i, (x, y) in enumerate(zip(dopant_pred_el, dopant_pred_hv))]
    
    # Assign the custom hover text to the figure
    fig.update_traces(text=hover_text, hoverinfo="text")

    # Add horizontal and vertical dashed lines
    fig.add_shape(
        type="line",
        x0=22.16,
        x1=22.16,
        y0=min(dopant_pred_hv),
        y1=600,
        line=dict(color="red", width=3, dash="dash"),
    )



    
    fig.add_shape(
        type="line",
        x0=min(dopant_pred_el),
        x1=32,
        y0=495.3,
        y1=495.3,
        line=dict(color="red", width=3, dash="dash"),
    )
    
    # Set the plot title
    fig.update_layout(title='Hardness-Elongation Synergy Optimization')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write(dopant_input)


# st.write(" NOTE: The elongation model is trained with the datasets characterized with the experimental parameters : cylindrical (compression test specimens) and rectangular geometries (tensile test specimens ), specimen dimension in the magnitude of several mm, strain rate in the range 1-10 X e-4 /s and, loading time of 10-15 s.")
# st.write("LIMITATION: This software app is validated only for ductile and highly ductile MPEAs. Not recommended for use in materials with ductility less than 5 % .")

st.markdown("<h2 style='color:red; font-size:24px; display: inline;'>How to Cite AlloyManufacturingNet:".format(alloy_style), unsafe_allow_html=True)
# st.write("How to Cite AlloyManufacturingNet ")
st.write("If you use AlloyManufacturingNet in your work, please cite: "
         "S. Poudel et al., AlloyManufacturingNet for discovery and design "
         "of hardness-elongation synergy in multi-principal element alloys"
        ", Engineering Applications of Artificial Intelligence, 132 (2024) 107902 ")

link='DOI: [https://doi.org/10.1016/j.engappai.2024.107902](https://doi.org/10.1016/j.engappai.2024.107902)'
st.markdown(link,unsafe_allow_html=True)

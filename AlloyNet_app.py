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
st.markdown('<h1 style="color:purple;">AlloyManufacturingNet </h1>', unsafe_allow_html=True)
# Add a dropdown to select a pre-defined formula
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: "plots/logo.png";
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
import streamlit as st
import pandas as pd

# Create two tabs in the sidebar
tab_options = ["New alloy design", "HV-EL Synergy Optimization"]
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
    from io import StringIO
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
    # Display the selected options
    # Define the text style for hardness and elongation with values and units
    alloy_style = "<h2 style='color:green; font-size:24px;'>{} </h2>".format(selected_predefined_formula)
    manufacturing_style = "<h2 style='color:green; font-size:24px;'>{} </h2>".format(selected_fabrication_type)
    # Display the styled text using st.markdown on the same line
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Multi-principal element alloy:</h2> {}".format(alloy_style), unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>Fabrication Type:</h2> {}".format(manufacturing_style), unsafe_allow_html=True)
    #st.write("Multi-principal element alloy:", selected_predefined_formula)
    #st.write("Fabrication Type:", selected_fabrication_type)
    # Display the selected options
    #st.write("Multi-principal element alloy:", selected_predefined_formula)
    #st.write("Fabrication Type:", selected_fabrication_type)
    # Define the text style for hardness and elongation with values and units
    hardness_style = "<h2 style='color:green; font-size:24px;'>{} HV</h2>".format(hardness)
    elongation_style = "<h2 style='color:green; font-size:24px;'>{} %</h2>".format(elongation)
    
    # Display the styled text using st.markdown on the same line
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>The hardness is:</h2> {}".format(hardness_style), unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue; font-size:24px; display: inline;'>The elongation is:</h2> {}".format(elongation_style), unsafe_allow_html=True)



    
    # Display other content as needed
    st.markdown("<h3 style='color:red;'>Composition-based feature vector:</h3>", unsafe_allow_html=True)
    # st.write(df_input_target)


    # Define the property names in LaTeX format
    # property_names_latex = [
    #     r"\text{Mean Atomic Radius difference } (\delta) ",
    #     r"\text{Electronegativity asymmetry } (\Delta \chi ",
    #     r"\text{Melting Temperature asymmetry } (\Delta T_m) ",
    #     r"\text{Average Melting Temperature } (T_{m}(K)) ",
    #     r"\text{Valence Electron Concentration } (VEC) ",
    #     r"\text{Average Atomic Number } (AN) ",
    #     r"\text{Thermal Conductivity } (K) ",
    #     r"\text{Average Bulk Modulus } (B) ",
    #     r"\text{Bulk Modulus Asymmetry } (\Delta B) ",
    #     r"\text{Average Shear Modulus } (G) ",
    #     r"\text{Shear Modulus Asymmetry } (\Delta G) ",
    #     r"\text{Entropy of Mixing asymmetry } (\Delta S_{mix}) ",
    #     r"\text{Geometrical Parameter } (\lambda) ",
    #     r"\text{Enthalpy of Mixing asymmetry } (\Delta H_{mix}) ",
    #     r"\text{Dimensionless parameter } (\Omega) ",
    # ]

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
   
    # # Display the property names and their corresponding values in LaTeX formatting
    # for i in range(len(property_names_latex)):
    #     st.latex("{} : {}".format(property_names_latex[i], values[i]))

    
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

if selected_tab == "Synergy Optimization":
    
    base_material_options = ["(VNbTa)", "(ZrHfNb)", "(MoNbTa)", "(CrFeCoNi)", "(CoCrNi)"]
    base_composition = st.sidebar.selectbox("Base MPEA", base_material_options)

    fabrication_type_options = ["CAST", "POWDER", "ANNEAL", "WROUGHT", "OTHER"]
    fab_type = st.sidebar.selectbox('Select Fabrication Type:', fabrication_type_options)

    first_dopants_options = ["Zr", "Mo", "W", "Ta",  "Hg", "Mo", "Ti"]
    first_dopant = st.sidebar.selectbox("First Dopants", first_dopants_options)

    second_dopants_options = ["W", "Ta", "Mg", "Ti", "Zr", "Hg"]
    second_dopant = st.sidebar.selectbox("Second Dopants", second_dopants_options)



    colorset_1 = ['Picnic', 'Viridis', 'Rainbow', 'Blackbody', 'Jet', 'Portland', 'Cividis', 'Electric']
    hv_colorset = st.sidebar.selectbox('Select Colorset for HV:', colorset_1)

    colorset_2 = ['Jet', 'Picnic', 'Viridis', 'Rainbow', 'Blackbody',  'Portland', 'Cividis', 'Electric']
    el_colorset = st.sidebar.selectbox('Select Colorset for EL:', colorset_2)
    
    # Perform actions or display content based on the selected options
    # st.write("Selected Base Piezo-material:", base_composition)
    # st.write("Selected First Dopant:", first_dopant)
    # st.write("Selected Second Dopant:", second_dopant)
    # Additional code for this tab
    pole_labels=[ base_composition,first_dopant,second_dopant]
    
    # st.dataframe(df_test, height=570)
    ternary_hv, dopant_input,df_alloy, dopant_pred_hv = ternary_plot(fab_cat=fab_type, pole_labels=[ base_composition,first_dopant,second_dopant],model_of='hardness', colorscale=hv_colorset)
    ternary_el, dopant_input,df_alloy, dopant_pred_el = ternary_plot(fab_cat=fab_type, pole_labels=[ base_composition,first_dopant,second_dopant],model_of='elongation', colorscale=el_colorset)

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
        x1=max(dopant_pred_el),
        y0=495.3,
        y1=495.3,
        line=dict(color="red", width=3, dash="dash"),
    )



    
    # Set the plot title
    fig.update_layout(title='Hardness-Elongation Synergy Optimization')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    
    st.write(dopant_input)

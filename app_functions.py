#!/usr/bin/env python
# coding: utf-8

# # 1. Preprocessing

# ## 1.1 Import Relevant Libraries and .CSV

# In[ ]:


def featurization(df):
    df = stc.featurize_dataframe(df, "Alloys",ignore_errors=True,return_errors=True)
    df = ef.featurize_dataframe(df, "composition",ignore_errors=True,return_errors=True)   
    return df


# In[ ]:


def properties_calculation(dataframe):
    # Input featurs calculation
    df = dataframe
    properties = []
    for number in range(len(df['Alloys'])):

        mpea = df['composition'][number]
        #print(Composition(mpea).as_dict().keys())
        element = list(Composition(mpea).as_dict().keys()) # List element present in Alloys ['Al', 'Cr', 'Fe', 'Ni', 'Mo']
        fraction_composition = list(Composition(mpea).as_dict().values()) # List Fraction composition of corresponding element in an Alloy eg. [1.0, 1.0, 1.0, 1.0, 1.0]
        total_mole = sum(fraction_composition) # Sum of elemental composition

        atomic_number = []
        bulk_modulus = []
        shear_modulus = []
        molar_heat = []
        thermal_conductivity = []
        mole_fraction = []
        X_i = []
        r_i = []
        Tm_i = []
        VEC_i= []
        R = 8.314

        for i in element:

            atomic_number.append(Element(i).Z)
            #molar_heat.append(Cp_dict[i])

            bulk_b =Element(i).bulk_modulus

            if type(bulk_b) == type(None):
                for j in bulk_modulus_b: bulk_b = (bulk_modulus_b.get(j))       

            bulk_modulus.append(bulk_b)

            #print(bulk_modulus)

            shear_g = (Element(i).rigidity_modulus)
            if type(shear_g) == type(None):
                for s in shear_modulus_g: shear_g = ((shear_modulus_g.get(s)))
            shear_modulus.append(shear_g)

            thermal_conductivity.append(Element(i).thermal_conductivity)
            mole_fraction.append(Composition(mpea).get_atomic_fraction(i)) # Calculates mole fraction of mpea using "Composition" functions

            X_i.append(Element(i).X) # Calculate individual electronegativity using "Element" function

            r_i.append(Element(i).atomic_radius) if Element(i).atomic_radius_calculated == None else r_i.append(Element(i).atomic_radius_calculated) # There are two functions present in Element␣class of pymatgen, so here checking using if conditional in both functions␣to not miss any value
            Tm_i.append(Element(i).melting_point) # Calculating melting point of every element using "Element" class and function
            try: VEC_i.append(DemlData().get_elemental_property(Element(i),"valence")) # VEC is also present in 2 locations in matminer, first is the␣function "DemlData()"
            except KeyError:
                if i in VEC_elements: VEC_i.append(float(VEC_elements.get(i))) #In case data is not present in "DemlData()" function, there is a csv file␣inside matminer opened earlier as "elem_prop_data" in the very first cell

        # Average Atomic Number
        AN = sum(np.multiply(mole_fraction, atomic_number))

        # Average Molar Heat coefficient
        #Cp_bar = sum(np.multiply(mole_fraction, molar_heat))    
        #print(Cp_bar)

        #term_Cp = (1-np.divide(molar_heat, Cp_bar))**2
        #del_Cp = sum(np.multiply(mole_fraction, term_Cp))**0.5 

        # Thermal Conductivity
        k = sum(np.multiply(mole_fraction, thermal_conductivity))

        # Bulk Modolus
        bulk = sum(np.multiply(mole_fraction, bulk_modulus)) # Bulk modulus of 'Zr' not present
        
        # Bulk modolus asymmetry
        term_bulk = (1-np.divide(bulk_modulus, bulk))**2
        del_bulk = sum(np.multiply(mole_fraction, term_bulk))**0.5         

        # Shear Modolus
        shear= sum(np.multiply(mole_fraction, shear_modulus))
        
        # Shear modolus asymmetry
        term_shear = (1-np.divide(shear_modulus, shear))**2
        del_shear = sum(np.multiply(mole_fraction, term_shear))**0.5         

        # Calculation of Atomic Radius Difference (del)

        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2
        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5 
        #print(number,element,mole_fraction,r_i,r_bar,term,atomic_size_difference)


        # Electronegativity (del_X)
        X_bar = sum(np.multiply(mole_fraction, X_i))
        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i,X_bar))**2)))**0.5
        #term_X = (1-np.divide(X_i, X_bar))**2
        #del_Chi = (sum(np.multiply(mole_fraction, term_X)))**0.5 

        # Difference Melting Temperature
        T_bar = sum(np.multiply(mole_fraction, Tm_i))
        del_Tm =(sum(np.multiply(mole_fraction, (np.subtract(Tm_i,T_bar))**2)))**0.5

        # Average Melting Temperature
        Tm = sum(np.multiply(mole_fraction, Tm_i))    

        # Valence Electron Concentration
        VEC = sum(np.multiply(mole_fraction, VEC_i))

        # Entropy of mixing
        del_Smix = -WenAlloys().compute_configuration_entropy(mole_fraction)*1000 #WenAlloys class imported from matminer library



        HEA = element
        #print(len(mole_fraction), len(HEA))


        # Enthalpy of mixing
        AB = []
        C_i_C_j = []
        del_Hab = []
        for item in range(len(HEA)):
            for jitem in range(item, len(HEA)-1):
                AB.append(HEA[item] + HEA[jitem+1])
                C_i_C_j.append(mole_fraction[item]*mole_fraction[jitem+1])
                del_Hab.append(round(Miedema().deltaH_chem([HEA[item], HEA[jitem+1]], [0.5, 0.5], 'ss'),3)) # Calculating binary entropy of mixing at 0.5-0.5␣ (equal) composition using Miedema class of "matminer" library

        omega = np.multiply(del_Hab, 4)
        del_Hmix = sum(np.multiply(omega, C_i_C_j))

        # Geometrical parameters
        lemda = np.divide(del_Smix, (atomic_size_difference)**2)
        #print(number,del_Smix,atomic_size_difference, lemda)

        parameter = Tm*del_Smix/abs(del_Hmix) 
        #print(number,"lemda, parameter", lemda, parameter)


        properties.append([len(element), " ".join(element), " ".join(list(map(str, fraction_composition))),total_mole, round(sum(mole_fraction),1), atomic_size_difference, round(del_Chi, 4),del_Tm, Tm, VEC, AN, k, bulk,del_bulk,shear,del_shear, round(del_Smix, 4),round(lemda,4), round(del_Hmix, 4),round(parameter,4)])
    prop_data = pd.DataFrame(properties, columns=['No of Components','Component','Moles of individual Components', 'Total Moles', 'Sum of individual MoleFractions', '$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K','B', 'ΔB','G', 'ΔG','ΔSmix','$\lambda$', 'ΔHmix','$\Omega$'])
    df = pd.concat([df, prop_data], axis = 1)
    
    df_input_target = df.iloc[:,[-20,-15,-14,-13,-12,-11,-10, -9, -8, -7, -6,-5, -4, -3, -2, -1,1]]
    
    return(df,df_input_target)


# In[ ]:


def r2_new(y_test,ideal_weighted_ensemble_prediction):
    import sklearn.metrics
    from sklearn.metrics import r2_score

    ideal_weighted_ensemble_prediction


    #Predict on test data
    y_test
    a=0.2 # Percentage error range
    r2_mean = r2_score(y_test,ideal_weighted_ensemble_prediction)
    plt.figure(figsize=(4,4.5),dpi=600)

    # plot x=y line 
    x_line = np.linspace(0, 1200, 1000)

    sns.lineplot(x=x_line, y=x_line,color='black',lw=0.75)

    print('Test R2 score: ', r2_mean)

    test_r2 = sns.regplot(x=y_test,y=ideal_weighted_ensemble_prediction,ci=None,scatter_kws=dict(s=8,color='r'),fit_reg=False)
    test_r2.set(title='Test $R^2 = $' +str(round(r2_mean,2))+' with 20% Error region')
    test_r2.set_xlabel("Real Test Targets", fontsize = 18)
    test_r2.set_ylabel("Predicted Test Value", fontsize = 18)

    Y1 = x_line*(1+a)
    Y2 = x_line*(1-a)

    sns.lineplot(x=x_line,y=Y1,lw=0.5,color='b',alpha=.2)
    sns.lineplot(x=x_line,y=Y2,lw=0.5,color='b',alpha=.2)

    plt.fill_between(x_line, Y1,x_line,color='b',alpha=.2)
    plt.fill_between(x_line, Y2,x_line,color='b',alpha=.2)

    #test_r2.figure.savefig('plots\\r2_test_hardness.pdf',dpi=1200)


# In[ ]:


def df_element_number(df):
    # Add a column of "Number of component" & "component" in each alloy system
    prop = []
    for number in range(len(df['Alloys'])):
        mpea = df['composition'][number]
        element = list(Composition(mpea).as_dict().keys()) # List element present in Alloys ['Al', 'Cr', 'Fe', 'Ni', 'Mo']
        prop.append([len(element), " ".join(element)])

        prop_data = pd.DataFrame(prop, columns=['No of Components', 'Component'])
    df = pd.concat([df, prop_data], axis = 1)
    return df


# In[ ]:


def element_number(df, fig_title='Hardness', fig_name='element_number_hardness'):
    import os
    import matplotlib
    import matplotlib.ticker as tck
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
    df['No of Components'].value_counts().plot(ax=ax, use_index=True, kind = 'bar')

    ax.tick_params(axis='x', labelrotation=0)
    plt.rcParams['font.size'] = '20'

    #plt.rcParams.update({'font.size': 20})
    plt.xlabel('Number of element in MPEA', fontsize=18)
    plt.ylabel('Count', fontsize=20)
    plt.title(fig_title,fontsize=20)

    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax.tick_params(bottom=True, left=True)
    #plt.tick_params(axis='both', which='both', length=3, width=2,color='black')
    
    plot_fig_loc = fig_name
    plt.savefig(plot_fig_loc,dpi=1200, bbox_inches='tight')
    plt.show()


# In[ ]:


def fab_type(df, fig_title='Hardness/ Elongation', fig_name='fab_type_hardness'):   
    import matplotlib.ticker as tck
    fig, ax = plt.subplots(figsize=(6,5),dpi=600)

    
    df['Fabrication_type'].replace(to_replace=["CAST"],
           value="CAT-A", inplace= True)
    
    df['Fabrication_type'].fillna("Unknown",inplace=True)
    
    df['Fabrication_type'].replace(to_replace=["OTHER","Unknown","WROUGHT"],
           value="CAT-B", inplace= True)    

    df['Fabrication_type'].replace(to_replace=["POWDER"],
           value="CAT-C", inplace= True) 
    
    df['Fabrication_type'].replace(to_replace=["ANNEAL"],
           value="CAT-D", inplace= True) 

    df['Fabrication_type'].value_counts(dropna=False).plot(ax=ax, use_index=True, kind = 'bar')    
    
    ax.tick_params(axis='x', labelrotation=20)
    #plt.xlabel('Fabrication route', fontsize=10)
    plt.ylabel('Count', fontsize=18)
    plt.rcParams.update({'font.size': 16})

    #ax.set_xticklabels( ('Cast', 'Unknown','Powder','Anneal','Other', 'Wrought') )

    plt.tick_params(axis='both', which='both', length=3, width=1,color='black')
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    #plt.ylim(0, 620)
    plt.title('Fabrication Processes ('+fig_title+')')
    plot_fig_loc = fig_name
    plt.savefig(plot_fig_loc,dpi=1200, bbox_inches='tight')
    plt.show()


# In[1]:


def phase_rearrange(df):
    df['Phase'].replace(to_replace=["FCC+Sec.", "FCC+B2+Laves",  "FCC+B2+Sec.", "FCC + Im", "FCC+L12+Sec", "FCC+B2","FCC+Laves", "FCC+L12+Sec.", "FCC+HCP+Sec." ],
           value="FCC+Im", inplace= True)

    df['Phase'].replace(to_replace=["BCC+Sec.","B2+BCC", "BCC+B2","BCC+B2+Laves", "BCC+Laves+Sec.", "BCC+Laves",  "BCC+B2+Sec.", "BCC + Im", "BCC + lm","BCC+B2+L12","BCC+BCC+Sec.", "BCC+Laves+Sec.", "BCC +Im"],
               value="BCC+Im", inplace= True)

    df['Phase'].replace(to_replace=["FCC+BCC", "FCC + BCC", "FCC+BCC+BCC", "BCC+FCC"],
               value="FCC+BCC", inplace= True)

    df['Phase'].replace(to_replace=["FCC+BCC+Sec.","FCC+BCC+B2","FCC+BCC+B2+Sec.", "FCC + BCC + Im", "FCC + BCC + B2"],
               value="FCC+BCC+Im", inplace= True)

    df['Phase'].replace(to_replace=["B2+Sec.","B2","Im", "FCC + BCC + Im"],
               value="FCC+BCC+Im", inplace= True)

    df['Phase'].replace(to_replace=["FCC+FCC", "FCC", "FCC+HCP", "FCC + HCP"],
               value="FCC", inplace= True)

    df['Phase'].replace(to_replace=["BCC+HCP", "BCC","BCC+BCC"],
               value="BCC", inplace= True)
    
    df['Phase'].replace(to_replace=["L12+B2", "B2+L12", ""],
               value="Im", inplace= True)

    df['Phase'].fillna("Unknown",inplace=True)

    df['Phase'].replace(to_replace=["Unknown",0, "Other"],
               value="Unknown", inplace= True)
    
    df['Phase'].value_counts(dropna=False)
    return df


# In[3]:


def phase_plot(df, fig_title='Hardness/ Elongation', fig_name='fab_type_hardness'):
    import seaborn as sns
    import matplotlib.ticker as tck
    sns.set(style="ticks", color_codes=True,font_scale=2.25)
    ax = plt.figure(figsize=(16,5),dpi=600)
    plt.tick_params(axis='x', labelrotation=30)
    sns.set(style='white',font_scale=2)
    sns.set_style('white', {'axes.linewidth': 0.5})
    plt.rcParams['xtick.major.size'] = 50
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True


    sns.boxplot(x=df['Phase'],y=df[fig_title], width= 0.4)
    plt.xticks(rotation=10)

    plt.rcParams.update({'font.size': 24})
    plt.tick_params(axis='both', which='both', length=3, width=1,color='black')
    ax.gca().yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    
    plot_fig_loc = fig_name
    plt.savefig(plot_fig_loc,dpi=1200, bbox_inches='tight')
    #plt.savefig('element_phase_hardness.png',dpi=1200)


# In[ ]:


def element_occurrence(df,limit_value=8, fig_title='Hardness/ Elongation', fig_name='element_occurrence'):
    
    df = df.loc[:, (df != 0).any(axis=0)]

    cols = list(df.columns.values)    #Make a list of all of the columns in the df
    set = df.astype(bool).sum(axis=0) # Extract the occurance of each element in the alloys

    element_df = set.to_frame()      # Convert extracted the occurance of each element in dataframe

    element_occurancy = element_df[limit_value:-3]
    element_occurancy.columns =['Occurrence']

    # Plotted shorted frequency of occurance
    #plt.figure(figsize=(30,6),dpi=600)
    element_occurancy.sort_values("Occurrence").plot(y='Occurrence', use_index=True, kind = 'bar') 
    plt.tick_params(axis='x', labelrotation=90)
    plt.ylabel('Count', fontsize=20)

    plt.rcParams.update({'font.size': 18})
    plt.tick_params(axis='both', which='both', length=3, width=1,color='black')
    plt.gca().yaxis.set_minor_locator(MultipleLocator(50))

    plt.rcParams['figure.figsize'] = [14,6]

    
    print(element_occurancy.sort_values("Occurrence"))
    print("\n The total number of datasets used for ML model is", element_df.iat[0,0])
    plt.xticks(rotation=10)
    #plt.savefig('element_occurance_hardness.png',dpi=1200)
    plot_fig_loc = fig_name
    plt.savefig(plot_fig_loc,dpi=1200, bbox_inches='tight')
    plt.show()


# In[ ]:


def data_elimination(df):
    # Eliminate dataset consisting less than 3 and greater than 8 element & alloys with fewer element occurance

    total_datasets = len(df.index)

    df = df[~df.Component.str.contains('Au')]
    df = df[~df.Component.str.contains('Nd')]
    df = df[~df.Component.str.contains('Ag')]
    #df = df[~df.Component.str.contains('Sc')]
    df = df[~df.Component.str.contains('Li')]
    df = df[~df.Component.str.contains('Mg')]

    df = df[~df.Component.str.contains('Re')]
    df = df[~df.Component.str.contains('Y')]
    #df = df[~df.Component.str.contains('Pd')]


    # Reset the index after deleting the rows.
    df = df.reset_index(drop=True)

    datasets_element = len(df.index)
    data_eliminated_by_element = total_datasets - datasets_element

    # Eliminate alloys with less than 3 elements and more than 8 elements
    df = df[(df['No of Components'] > 2) & (df['No of Components'] < 8)]

    # Reset the index after deleting the rows.
    df = df.reset_index(drop=True)

    print("Total Datasets", total_datasets)
    print("Datasets eleminated due to fewer number of element occurance:", data_eliminated_by_element)
    print("Datasets eleminated due to number of element in alloy:", datasets_element -len(df.index))
    print("Total Datasets for the hardness prediction", len(df.index))
    return(df)


# In[4]:


def fab_encoding(df,path='hardness_model_files\\'):
    df.Fabrication_type = df.Fabrication_type.fillna('OTHER')
    df.Fabrication_type = df.Fabrication_type.replace(to_replace=["WROUGHT"],
               value="OTHER")
    
    print(df['Fabrication_type'].value_counts(dropna=False))
    
    from sklearn.preprocessing import OneHotEncoder
    # creating one hot encoder object 
    enc = OneHotEncoder()
    fab_type = pd.DataFrame(enc.fit_transform(df[['Fabrication_type']]).toarray())
    
    import pickle
    pickle.dump(enc, open(path+'fab_encoding.pkl','wb'))
    
    return(fab_type, enc)
    


# In[ ]:


def input_target(datasets, fab_type, input_name):
    inputs = np.column_stack((fab_type,datasets))
    inputs = inputs.astype(float)

    df_all = pd.DataFrame(inputs, columns = ['Fab_1','Fab_2','Fab_3','Fab_4','No of Components']+input_name+["HV"])

    df_inputs = df_all.drop(['HV'], axis=1)
    df_targets = df_all['HV']
    return (df_all, df_inputs, df_targets)
    
def train_test_split(datasets, fab_type, input_name):
    
    df_all, df_inputs, df_targets = input_target(datasets, fab_type, input_name)
    # Split dataset in train-test

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_inputs, df_targets, test_size=0.1, random_state=42)

    #X_train, X_test,  = train_test_split(df_inputs, test_size=0.1, random_state=0)

    X_train_no_fab = X_train.drop(['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4','No of Components'], axis=1)
    X_train_fab = X_train.loc[:,['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4']]

    X_test_no_fab = X_test.drop(['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4','No of Components'], axis=1)
    X_test_fab = X_test.loc[:,['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4']]
    
    n_component = X_train.loc[:,['No of Components']]

    input_df = pd.concat([X_train_no_fab, y_train], axis=1)
    
    return(X_train_no_fab, X_train_fab, X_test_no_fab, X_test_fab,y_train, y_test,n_component)


# In[ ]:


def data_distribution(data,text,test_annotate=0,test_annotate2=0,limit=1200,distance=0.001,ylabel = 'Hardness (HV)', title="Hardness Distribution",plot_path="plots\\hardness\\"):
    
    import seaborn as sns
    import matplotlib.ticker as tck
    
    mean=data.mean()
    median=np.median(data)
    ten_per = np.percentile(data, 10)
    ninety_per = np.percentile(data, 90)
    print(text,'\n Mean: ',mean,'\n Median: ',median, '\n 10 Percentile:',ten_per, '\n 90 Percentile:', ninety_per)
    print("Number of ",text, data.shape[0])
    print("----------------------")
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    
    fig , ax = plt.subplots(figsize=(5,6), dpi=400)
    sns.kdeplot( y=data, color="dodgerblue",lw=0.75, shade=True, bw_adjust=1)
    
    # Plot Mean and Median
    plt.plot(distance,mean, marker="x", markersize=8, markeredgecolor="k", markerfacecolor="k", label="Mean",linestyle = 'None')
    plt.plot(distance,median, marker="^", markersize=6, markeredgecolor="blue", markerfacecolor="blue", label="Median",linestyle = 'None')
    
    # Plot 10 and 90 percentile
    plt.plot(distance,ten_per, marker="o", markersize=5, markeredgecolor="black", label="10% Percentile",linestyle = 'None')
    plt.plot(distance,ninety_per, marker="o", markersize=5, markeredgecolor="blue", label="90% Percentile",linestyle = 'None')

    x_values = [distance, distance]
    y_values = [ten_per, ninety_per]
    plt.plot(x_values, y_values, 'green', linestyle="-")
    
    # Annotations
    plt.text(distance*1.06, ten_per-test_annotate, str(int(ten_per))+ " (10%)", horizontalalignment='left', size='medium', color='black')
    plt.text(distance*1.06, ninety_per+test_annotate2, str(int(ninety_per))+ " (90%)", horizontalalignment='left', size='medium', color='black')

    #plt.gca().axes.get_xaxis().set_visible(False) # Remove x-axis lable
    
    plt.ylim(0, limit)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    
    plt.title(title+str(text)+" ("+str(data.shape[0])+" data)", pad=20, color='blue')
    plt.ylabel(ylabel,size=18)
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Number of x ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    
    # Crop shaded region above max and below min value
    plt.axhspan(0,min(data), color='white')
    plt.axhspan(max(data),limit, color='white')

    
    plt.legend(frameon=False,loc='upper right',fontsize=14)
    plt.savefig(plot_path+str(text)+ylabel+'_split.png',dpi=1200, bbox_inches='tight')
    #plt.legend(frameon=False);


# In[ ]:


def std_data(X_train_no_fab):
    # Standarize the input features
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    scaler = StandardScaler()
    robust = RobustScaler()
    #robust_X_train = robust.fit_transform(X_train_no_fab)
    #scaler.fit(X_train_no_fab)
    std_X_train = scaler.fit_transform(X_train_no_fab)

    std_df = pd.DataFrame(data=std_X_train, columns=input_name)
    #std_df['Hardness (HV)'] = y_train
    
    return(scaler, std_df)


# In[1]:


def heatmap(std_train_df,name,prop='HV'):
    import seaborn as sns
    sns.set(style="ticks", color_codes=True,font_scale=2.2)
    plt.figure(figsize=(22,12))
    cmap = sns.diverging_palette(133,10,s=80, l=55, n=9, as_cmap=True)
    cor_train = std_train_df.corr()
    sns.heatmap(cor_train, annot=True, fmt='.2f',cmap=cmap) #

    plt.savefig(name+'_pcc_all.pdf',dpi=1200,bbox_inches='tight')
    plt.show()

def pcc_fs(std_df,y_train,input_pcc,name,prop='HV'):
    
    std_all_feature = np.column_stack((std_df,y_train))
    std_train_df=pd.DataFrame(data=std_all_feature, columns=input_name+[prop])
    heatmap(std_train_df,name)
    
    X_train_pcc = std_train_df.loc[:,input_pcc+[prop]]
    import seaborn as sns
    plt.figure(figsize=(13,6))
    cor_mid = X_train_pcc.corr()
    sns.heatmap(cor_mid, annot=True, cmap= plt.cm.CMRmap_r,fmt='.2f')
    plt.savefig(name+'_pcc_fs.png',dpi=1200,bbox_inches='tight')
    plt.show()
    


# In[ ]:


def vif_value(datasets):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(datasets.values,i) for i in range(datasets.shape[1])]

    vif['features'] = datasets.columns

    return(vif)


# In[ ]:


def pairplot(std_df_pcc,n_component,name, plot_loc):
    import seaborn as sns
    df_pp = std_df_pcc.astype(float)
 
    #print("n_component:", n_component.value_counts(dropna=False))    
    n_component['No of Components'].replace(to_replace=[2,3,4],
        value="MEA", inplace= True)
    
    n_component['No of Components'].replace(to_replace=[5, 6, 7,8,9],
        value="HEA", inplace= True)
    
    
    #print("n_component:", n_component.value_counts(dropna=False))

    df_pp[' '] = n_component
    #print("....")

    #print("df_pp",df_pp[' '].value_counts(dropna=False))

    sns.set(style="ticks", color_codes=True,font_scale=2, palette= 'dark')
    
    g = sns.pairplot(df_pp, hue=' ', markers = ['o', "*"],plot_kws={'alpha':0.8,"s": 10,'edgecolor': 'none'})
 
    g.fig.set_size_inches(19,11)
    plt.title(name, y=-1.25,x=-3,fontsize=36)

    plt.savefig(plot_loc+name+'_pc_pairplot.pdf',dpi=1200,bbox_inches='tight')
    plt.show()


# In[ ]:


def pca_fs(std_df,name, title="a) Hardness PCA-1"):
    
    import seaborn as sns
    import matplotlib.ticker as tck
    
    # Plot PCA graph
    from sklearn.decomposition import PCA
    pca = PCA()
    sns.set(style="ticks", color_codes=True,font_scale=3)
    principalComponents = pca.fit_transform(std_df)
    plt.figure(figsize=(8,7))
    #plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_),color='purple', linewidth=3)
    plt.xlabel('No. of Principal Components')
    plt.ylabel('Cumulative EV')
    plt.title(title+ ' : Explained Variance ',color='blue', pad=20)
    
    plt.grid(alpha=0.75)
    plt.grid(True, which='minor')

    x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    #x=[2,4,6,8,10,12,14]
    plt.xlim(0, 14)
    values = range(len(x))
    plt.xticks(values, x)
    #plt.xticks()
    
    # Number of x ticks
    plt.xticks(range(0,14,2),rotation=0)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    

    plt.rcParams.update({'font.size': 30})
    plt.tick_params(axis='both', which='both', length=3, width=1,color='black')
    
    import numpy
    plt.yticks(numpy.linspace(0.3, 0.9, num=4))
    
    plt.ylim(0.2, 1.05)
    #plt.yticks(range(0.3,0.9,0.3),rotation=0)
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    sns.set(style="ticks", color_codes=True,font_scale=2)

    plt.savefig(name+'_fs.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    # Principal components to capture 0.9 variance in data
    pca_1 = PCA(0.9)
    df_pca = pca_1.fit_transform(std_df)
    
    comp = pca_1.n_components_
    print('No. of components for PCA:' ,comp)
    
    pca_new = PCA(n_components = comp)
    df_pca_new = pca_new.fit_transform(std_df)
    
    print('Explained variance for ', comp, 'components: ',pca_new.explained_variance_ratio_)
    print('Cumulative:', np.cumsum(pca_new.explained_variance_ratio_))
    
    return(pca_1,df_pca)


# In[ ]:


# Machine Learning Model



# The value used in the function plays no role as the different hyperparameter value will be used while calling "create_model" function
def create_model(lyrs=6, neuron_size=64, act='selu', opt='Adam', dr=0.0, learning_rate=0.001,init_weights= 'he_uniform', weight_constraint = 3):
    import tensorflow as tf

    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Dropout
    from tensorflow.keras.constraints import max_norm

    import numpy as np
    import matplotlib.pyplot as plt
    
    # clear model
    tf.keras.backend.clear_session()

    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(neuron_size,input_dim=input_dim, activation=act, kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))
    #tf.keras.layers.BatchNormalization(),
    
    # create additional hidden layers
    for i in range(1,lyrs):
        model.add(Dense(neuron_size, activation=act, kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
        model.add(Dropout(dr))
        
        #tf.keras.layers.BatchNormalization(),
    # add dropout, default is none
    #model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1, activation='ReLU'))  # output layer
    opt = Adam(learning_rate=learning_rate)
    huber = tf.keras.losses.Huber(delta=1.5)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=['mse', 'mape','mae',tf.keras.metrics.RootMeanSquaredError()])
    
    return model


# In[ ]:


def matrics_plot(history,train_matrics,val_matrics,lable_name,model_name, data_of,plot_path):
    import matplotlib.ticker as tck
    all_train_mae_histories = []
    train_mae_history = train_matrics
    all_train_mae_histories.append(train_mae_history)
    average_train_mae_history = [
        np.mean([x[i] for x in all_train_mae_histories]) for i in range(max_epochs)]

    all_val_mae_histories = []
    val_mae_history = val_matrics
    all_val_mae_histories.append(val_mae_history)
    average_val_mae_history = [
        np.mean([x[i] for x in all_val_mae_histories]) for i in range(max_epochs)]
    
    loss = train_matrics
    val_loss = val_matrics
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', linewidth=2, label='Training '+lable_name)
    plt.plot(epochs, val_loss, '--r',  linewidth=1, label='Validation '+lable_name)
    plt.title('Training and Validation '+lable_name)
    plt.xlabel('Epochs')
    plt.ylabel(lable_name)
    plt.legend()
    #plt.savefig('mae_hardness.pdf',dpi=1200)
    plt.show()
    
    def smooth_curve(points, factor=0.9):
      smoothed_points = []
      for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
      return smoothed_points

    smooth_train_mae_history = smooth_curve(average_train_mae_history[5:])
    smooth_val_mae_history = smooth_curve(average_val_mae_history[5:])
    
    sns.set(style="ticks", color_codes=True,font_scale=2.25)
    fig, ax = plt.subplots(figsize=(6,5.5),dpi=600)
    #plt.ylim(20, 120)    # y-label range
    plt.gca().yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    plt.plot(range(1, len(smooth_train_mae_history) + 1), smooth_train_mae_history, 'b', label = 'Training '+lable_name)
    plt.plot(range(1, len(smooth_val_mae_history) + 1),smooth_val_mae_history, '--r', label = 'Validation '+lable_name)
    plt.xlabel('Epochs')
    plt.ylabel(''+lable_name)
    #plt.title('Smooth Training and Validation '+lable_name)
    plt.title(data_of+': '+model_name)
    plt.legend()
    plt.savefig(plot_path+data_of+'_'+lable_name+'.pdf',dpi=1200, bbox_inches='tight')
    plt.show()


# In[ ]:


#Predict on test data

def r2_plot(model,input_datasets,target_datasets,name,model_name,plot_path="plots\\hardness\\_"):
    sns.set(style="ticks", color_codes=True,font_scale=1.25)
    a=0.2 # Percentage error range
    predictions_datasets = model.predict(input_datasets)

    import sklearn.metrics
    from sklearn.metrics import r2_score
    r2_test = r2_score(target_datasets, predictions_datasets)
    plt.figure(figsize=(4,4.2),dpi=600)

    # plot x=y line 
    x_line = np.linspace(0, 1000, 1000)
    
    sns.lineplot(x=x_line, y=x_line,color='black',lw=0.75)

    print('Test R2 score: ', r2_test)

    test_r2 = sns.regplot(x=target_datasets,y=predictions_datasets,ci=None,scatter_kws=dict(s=8,color='r'),fit_reg=False)
    test_r2.set(title=str(model_name)+'Performance on Test data,'+' $R^2$ = ' +str(round(r2_test,3)))
    test_r2.set_xlabel("Real Targets"+"("+name+")", fontsize = 16)
    test_r2.set_ylabel("Predicted Value"+"("+name+")", fontsize = 16)

    Y1 = x_line*(1+a)
    Y2 = x_line*(1-a)

    sns.lineplot(x=x_line,y=Y1,lw=0.5,color='b',alpha=.2)
    sns.lineplot(x=x_line,y=Y2,lw=0.5,color='b',alpha=.2)

    test_r2.fill_between(x_line, Y1,x_line,color='b',alpha=.2)
    test_r2.fill_between(x_line, Y2,x_line,color='b',alpha=.2)
    
    # x and y ticks
    listOf_Yticks = np.arange(0, 1400, 200)
    plt.yticks(listOf_Yticks)
    plt.xticks(listOf_Yticks)
    
    
    
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))


    test_r2.figure.savefig('r2_hardness_'+str(name)+'.png',dpi=1200, bbox_inches='tight')


# In[ ]:


def ensemble_model(models, test_datasets,y_test):    
    import sklearn.metrics
    from sklearn.metrics import r2_score
    preds_array=[]
    #type(preds_df)
    for i in range(len(models)):
        preds = models[i].predict(test_datasets[i])
        r2_mean = r2_score(y_test,preds)
        preds_array.append(preds)

        print("R sq. for Model",i,r2_mean)
        #print(preds)
    preds_array=np.array(preds_array)    
    summed = np.sum(preds_array, axis=0)
    ensemble_prediction = np.argmax(summed, axis=1)
    mean_preds = np.mean(preds_array, axis=0)

    r2_mean = r2_score(y_test,mean_preds)
    print("Average:",r2_mean )


    # Weight calculations
    df = pd.DataFrame([])

    for w1 in range(0, 4):
        for w2 in range(0,4):
            for w3 in range(0,4):
                for w4 in range(0,4):
                    wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                    wted_preds1 = np.tensordot(preds_array, wts, axes=((0),(0)))
                    wted_ensemble_pred = np.mean(wted_preds1, axis=1)
                    weighted_r2 = r2_score(y_test, wted_ensemble_pred)
                    df = pd.concat([df,pd.DataFrame({'acc':weighted_r2,'wt1':wts[0],'wt2':wts[1], 
                                                 'wt3':wts[2],'wt4':wts[3] }, index=[0])], ignore_index=True)

    max_r2_row = df.iloc[df['acc'].idxmax()]
    print("Max $R^2$ of ", max_r2_row[0], " obained with w1=", max_r2_row[1]," w2=", max_r2_row[2], " w3=", max_r2_row[3], " and w4=", max_r2_row[4])  
    return(preds_array)


# In[ ]:


def weighted_ensemble(preds_array, ideal_weights,y_test,plot_path="plots\\hardness\\", plot_title="Hardness Ensemble Model",limit_value1=1200,limit_value2=1050,value_gap=200):    
    import sklearn.metrics
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
    from sklearn.metrics import r2_score
    
    import matplotlib.ticker as tck
    #Use tensordot to sum the products of all elements over specified axes.

    ideal_weighted_preds = np.tensordot(preds_array, ideal_weights, axes=((0),(0)))
    ideal_weighted_ensemble_prediction = np.mean(ideal_weighted_preds, axis=1)
    
    
    mae = mean_absolute_error(y_test, ideal_weighted_ensemble_prediction)
    mse = mean_squared_error(y_test, ideal_weighted_ensemble_prediction)
    rmse = np.sqrt(mse) # or mse**(0.5)  
    mape = mean_absolute_percentage_error(y_test, ideal_weighted_ensemble_prediction)
    r2 = r2_score(y_test, ideal_weighted_ensemble_prediction)

    print("Results of sklearn.metrics:")
    print("MAE:",mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R-Squared:", r2)


    #Predict on test data
    a=0.2 # Percentage error range
    r2_mean = r2_score(y_test,ideal_weighted_ensemble_prediction)
    plt.figure(figsize=(6.25,6),dpi=600)

    # plot x=y line 
    x_line = np.linspace(0, limit_value1, 2)
    

    sns.lineplot(x=x_line, y=x_line,color='black',lw=0.75)

    print('Test R2 score: ', r2_mean)

    test_r2 = sns.regplot(x=y_test,y=ideal_weighted_ensemble_prediction,ci=None,scatter_kws=dict(s=8,color='b'),fit_reg=False)
    test_r2.set_title('$R^2 = $' +str(round(r2_mean,2))+plot_title, fontsize = 24)
    test_r2.set_xlabel("Real Test Targets", fontsize = 24)
    test_r2.set_ylabel("Predicted Test Value", fontsize = 24)

    Y1 = x_line*(1+a)
    Y2 = x_line*(1-a)
    
    

    plt.ylim(0, limit_value2)
    plt.xlim(0, limit_value2)
    test_r2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    test_r2.xaxis.set_minor_locator(tck.AutoMinorLocator(2))

    sns.lineplot(x=x_line,y=Y1,lw=0.5,color='r',alpha=.2)
    sns.lineplot(x=x_line,y=Y2,lw=0.5,color='r',alpha=.2)

    plt.fill_between(x_line, Y1,x_line,color='r',alpha=.2)
    plt.fill_between(x_line, Y2,x_line,color='r',alpha=.2)
    test_r2.set(xticks=np.arange(0, limit_value1, value_gap))
    test_r2.set(yticks=np.arange(0, limit_value1, value_gap))
    test_r2.figure.savefig(plot_path+plot_title+'_r2.png',dpi=1200, bbox_inches='tight')


# In[1]:


def prediction_model(df,input_pcc,prop="Hardness (HV)", path='hardness_model_files\\'):    
    input_name = ['$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K', 'B', 'ΔB', 'G', 'ΔG','ΔSmix','$\lambda$', 'ΔHmix','$\Omega$']
    datasets = df.iloc[:,[-15,-14,-13,-12,-11,-10, -9, -8, -7, -6,-5, -4, -3, -2, -1]]

    ##############################################
    # One-hot encoding transform from train-model
    ###############################################

    
    import pickle
    enc = pickle.load(open(path+'fab_encoding.pkl','rb'))
    # Cluster WROUGHT and Unknown into Other
    
    fab_type = pd.DataFrame(enc.transform(df[['Fabrication_type']]).toarray())

    ###################################################
    # Input feature Standarization
    ################################################
    import pickle
    scaler = pickle.load(open(path+'scaler.pkl','rb'))

    std_input = scaler.transform(datasets)

    std_df = pd.DataFrame(data=std_input, columns=input_name)


    #########################################################
    # Feature Selected
    ############################################

    # PCC_1

    df_pcc_1 = std_df.loc[:,input_pcc]

    # PCC_2

    inputs_all_fab = np.column_stack((df_pcc_1, fab_type))
    df_pcc_2= inputs_all_fab.astype(float)


    # PCA-1
    pca_1 = pickle.load(open(path+'pca_1.pkl','rb'))
    df_pca_1 = pca_1.transform(std_df)
    df_pca_1 = pd.DataFrame(df_pca_1)

    # PCA-2
    inputs_all_fab_pca = np.column_stack((std_df, fab_type))
    inputs_all_fab_pca = inputs_all_fab_pca.astype(float)
    
    pca_2 = pickle.load(open(path+'pca_2.pkl','rb'))
    df_pca_2 = pca_2.transform(inputs_all_fab_pca)
    df_pca_2 = pd.DataFrame(df_pca_2)


    #########################################
    # Load ML Model
    #########################################
    from keras.models import load_model

    load_model_pcc_1 = load_model(path+'best_model_pcc_1.h5')
    
    load_model_pcc_2 = load_model(path+'best_model_pcc_2.h5')
    load_model_pca_1 = load_model(path+'best_model_pca_1.h5')
    load_model_pca_2 = load_model(path+'best_model_pca_2.h5')


    ########################################
    # Ensemble Model
    #######################################
    models = [load_model_pcc_1,load_model_pcc_2, load_model_pca_1, load_model_pca_2]
    test_datasets = [df_pcc_1,df_pcc_2, df_pca_1, df_pca_2]

    ideal_weights = [0.1,0.3, 0.3, 0.3] 
    
    
 
    #preds_array = ensemble_model(load_models, test_datasets, y_test)
    #weighted_ensemble(preds_array, ideal_weights,y_test,plot_path="plots\\hardness\\", plot_title="Hardness Ensemble Model",limit_value1=1200,limit_value2=1050,value_gap=200)
    
#    '''    
    y_test = df[prop]  
    import sklearn.metrics
    from sklearn.metrics import r2_score
    preds_array=[]
 
    #type(preds_df)
    for i in range(len(models)):
        preds = models[i].predict(test_datasets[i])
        #r2_mean = r2_score(y_test,preds)
        preds_array.append(preds)

        #print(i,r2_mean)
        #print(preds)
    preds_array=np.array(preds_array)    
    summed = np.sum(preds_array, axis=0)
    ensemble_prediction = np.argmax(summed, axis=1)
    mean_preds = np.mean(preds_array, axis=0)



    # Weight calculations
    df = pd.DataFrame([])

    for w1 in range(0, 4):
        for w2 in range(0,4):
            for w3 in range(0,4):
                for w4 in range(0,4):
                    wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                    wted_preds1 = np.tensordot(preds_array, wts, axes=((0),(0)))
                    wted_ensemble_pred = np.mean(wted_preds1, axis=1)
                    weighted_r2 = r2_score(y_test, wted_ensemble_pred)
                    df = pd.concat([df,pd.DataFrame({'acc':weighted_r2,'wt1':wts[0],'wt2':wts[1], 
                                                 'wt3':wts[2],'wt4':wts[3] }, index=[0])], ignore_index=True)

    max_r2_row = df.iloc[df['acc'].idxmax()]
    print("Max $R^2$ of ", max_r2_row[0], " obained with w1=", max_r2_row[1]," w2=", max_r2_row[2], " w3=", max_r2_row[3], " and w4=", max_r2_row[4]) 
#    '''    
 
    
    #Use tensordot to sum the products of all elements over specified axes.
    ideal_weighted_preds = np.tensordot(preds_array, ideal_weights, axes=((0),(0)))
    ideal_weighted_ensemble_prediction = np.mean(ideal_weighted_preds, axis=1)
    
    return ideal_weighted_ensemble_prediction


# In[ ]:

#########################################################################################################
def prediction_model_new(df, predict='hardness'): 
    # For hardness
    input_pcc = ['$\delta$', 'Δ$\chi$', 'ΔTm', 'VEC', 'ΔB', 'ΔG', '$\lambda$', 'ΔHmix']
    path='hardness_model_files\\'
    ideal_weights = [0.3,0.3,0.1,0.3]
    
    input_name = ['$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K', 'B', 'ΔB', 'G', 'ΔG','ΔSmix','$\lambda$', 'ΔHmix','$\Omega$']
    datasets = df.iloc[:,[-15,-14,-13,-12,-11,-10, -9, -8, -7, -6,-5, -4, -3, -2, -1]]

    ##############################################
    # One-hot encoding transform from train-model
    ###############################################
    
    import pickle
    enc = pickle.load(open(path+'fab_encoding.pkl','rb'))
    # Cluster WROUGHT and Unknown into Other
    
    fab_type = pd.DataFrame(enc.transform(df[['Fabrication_type']]).toarray())

    ###################################################
    # Input feature Standarization
    ################################################
    import pickle
    scaler = pickle.load(open(path+'scaler.pkl','rb'))

    std_input = scaler.transform(datasets)

    std_df = pd.DataFrame(data=std_input, columns=input_name)

    #########################################################
    # Feature Selected
    ############################################

    # PCC_1
    #input_pcc = ['$\delta$', 'Δ$\chi$', 'ΔTm', 'VEC', '$\lambda$', 'ΔHmix']
    df_pcc_1 = std_df.loc[:,input_pcc]

    # PCC_2

    inputs_all_fab = np.column_stack((df_pcc_1, fab_type))
    df_pcc_2= inputs_all_fab.astype(float)


    # PCA-1
    pca_1 = pickle.load(open(path+'pca_1.pkl','rb'))
    df_pca_1 = pca_1.transform(std_df)
    df_pca_1 = pd.DataFrame(df_pca_1)

    # PCA-2
    inputs_all_fab_pca = np.column_stack((std_df, fab_type))
    inputs_all_fab_pca = inputs_all_fab_pca.astype(float)
    
    pca_2 = pickle.load(open(path+'pca_2.pkl','rb'))
    df_pca_2 = pca_2.transform(inputs_all_fab_pca)
    df_pca_2 = pd.DataFrame(df_pca_2)


    #########################################
    # Load ML Model
    #########################################
    from keras.models import load_model

    load_model_pcc_1 = load_model(path+'best_model_pcc_1.h5')
    
    load_model_pcc_2 = load_model(path+'best_model_pcc_2.h5')
    load_model_pca_1 = load_model(path+'best_model_pca_1.h5')
    load_model_pca_2 = load_model(path+'best_model_pca_2.h5')

    ########################################
    # Ensemble Model
    #######################################
    models = [load_model_pcc_1,load_model_pcc_2, load_model_pca_1, load_model_pca_2]
    test_datasets = [df_pcc_1,df_pcc_2, df_pca_1, df_pca_2]

    #ideal_weights = [0.3,0.3, 0.1, 0.3] 
    
    #preds_array = ensemble_model(load_models, test_datasets, y_test)
    #weighted_ensemble(preds_array, ideal_weights,y_test,plot_path="plots\\hardness\\", plot_title="Hardness Ensemble Model",limit_value1=1200,limit_value2=1050,value_gap=200)
    
    import sklearn.metrics
    from sklearn.metrics import r2_score
    preds_array=[]
 
    #type(preds_df)
    for i in range(len(models)):
        preds = models[i].predict(test_datasets[i])
        #r2_mean = r2_score(y_test,preds)
        preds_array.append(preds)

        #print(i,r2_mean)
        #print(preds)
    preds_array=np.array(preds_array)    
    summed = np.sum(preds_array, axis=0)
    ensemble_prediction = np.argmax(summed, axis=1)
    mean_preds = np.mean(preds_array, axis=0)
 
    #Use tensordot to sum the products of all elements over specified axes.
    ideal_weighted_preds = np.tensordot(preds_array, ideal_weights, axes=((0),(0)))
    ideal_weighted_ensemble_prediction = np.mean(ideal_weighted_preds, axis=1)
    
    return ideal_weighted_ensemble_prediction
#########################################################################################################################################

# In[ ]:


def ternary_plot(df,input_pcc,ideal_weights,element1,element2,fab_cat="CAT-A", pole_labels=['Al','Ti', '(CrFeNi)'],model_of='hv',colorscale='Jet'):
    df = df.set_axis(['Alloys'], axis=1, inplace=False)

    df["Fabrication_type"]=fab_cat
    
    df = featurization(df)
    #colorscale='Jet'
    if model_of=="hv":
        model_path='hardness_model_files\\'
        ticksuffix=""
        #colorscale='Blackbody'
        baxis_min=0.1
    else:
        model_path='elongation_model_files\\'
        ticksuffix="%"
        #colorscale="Jet"
        baxis_min=0.1
        


    df_prop, df_input_target = properties_calculation(df)
    
    #print(df_plot.head())
    predicted_value =prediction_model_new(df_prop,input_pcc,ideal_weights, path=model_path)
    df_result_ternary = pd.DataFrame(predicted_value)
    
    import plotly.figure_factory as ff
    import numpy as np

    #Al = np.array([0,0,0,0,0,0,0,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
    #Ti = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0, 0,0,0,0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1])
    comp = 1 - element1 - element2

    print(predicted_value.shape,element1.shape,comp.shape)

    fig = ff.create_ternary_contour(np.array([comp,element1,element2]), predicted_value,
                                    pole_labels=pole_labels,
                                    interp_mode='cartesian',
                                    ncontours=12,
                                    colorscale=colorscale,
                                    showscale=True,
                                    width=500, height=400)

    fig.update_ternaries(baxis_nticks=5)
    fig.update_ternaries(aaxis_nticks=5)
    fig.update_ternaries(caxis_nticks=5)

    fig.update_layout(
        #title=title,
        title_font_size=20,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        legend_title="Legend Title",
        #value_title = "a"
        font=dict(
            #family="Courier New, monospace",
            size=24,
            color="black"
        )
    )
    fig.update_ternaries(sum=1, baxis_min=baxis_min);

    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=0.05, x=1, 
                  ticks="outside", ticksuffix=ticksuffix))

    # Axis labels. (See below for corner labels.)
    fontsize = 28
    offset = 0.08

    plot_fig_loc = ('plots\\predictions\\Test_ternary_plot.pdf')

    #plt.savefig('plots\\predictions\\Test_ternary_plot.pdf')
    fig.show()

    return predicted_value
    
##########################################################################################


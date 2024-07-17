# Essential libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
from streamlit_option_menu import option_menu
import re

#set up page configuration for streamlit

icon='https://st2.depositphotos.com/1000128/7250/i/450/depositphotos_72503649-stock-photo-copper-pipes.jpg'
st.set_page_config(page_title='INDUSTRIAL COPPER MODELING',page_icon=icon,initial_sidebar_state='expanded',
                        layout='wide',menu_items={"about":'This streamlit application was developed by N.P.Nivesh'})

st.title("⚙️:rainbow[INDUSTRIAL COPPER MODELING]🏭")

#set up the sidebar with optionmenu

with st.sidebar:
    selected = option_menu("MainMenu",
                            options=["Home","Predictions","About"],
                            icons=["house","lightbulb","info-circle"],
                            default_index=1,
                            orientation="vertical",)
    
# set up the information for 'Home' menu

if selected == 'Home':
    title_text = '''<h1 style='font-size: 30px;text-align: center;'>INDUSTRIAL COPPER</h1>'''
    st.markdown(title_text, unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.subheader(':blue[What is copper?]')

        st.markdown('''<h5 style='color:grey;font-size:21px'> Copper is a reddish brown metal that is found in abundance all around the world, 
                    while the top three producers are Chile, Peru, and China. Historically, copper was the first metal to be worked 
                    by human hands. When we discovered that it could be hardened with tin to make bronze around 3000 BC,
                    the Bronze Age was ushered in, changing the course of humanity.''',unsafe_allow_html=True)

    with col2:
        st.image('https://st2.depositphotos.com/1000128/7250/i/450/depositphotos_72503649-stock-photo-copper-pipes.jpg')

    st.subheader(':blue[What Is Copper Used For?]')
    st.markdown('''<h5 style='color:grey;font-size:20px'>According to the Copper Development Association (CDA) there 
                are four different areas of industry where copper is utilized:<br>
                - Electrical: 65% <br>
                - Construction: 25% <br>
                - Transport: 7% <br>
                - Other: 3% ''',unsafe_allow_html=True)
    with st.container():
        with st.expander(':blue[***Electrical Copper***]'):
            st.markdown('''<h6 style='color:grey;font-size:18px'>Copper is used in virtually all electrical wiring (except for power lines, 
                        which are made with aluminum) because it is the second most electrically conductive metal aside from silver 
                        which is much more expensive. In addition to being widely available and inexpensive, it is malleable and easy to
                        stretch out into very thin, flexible but strong wires, making it ideal to use in electrical infrastructure.<br>Aside from 
                        electrical wiring,copper is also used in heating elements, motors, renewable energy, internet lines, and electronics.
                        ''',unsafe_allow_html=True)
    with st.container():
        with st.expander(':blue[***Copper for Construction, Piping, & Design***]'):
            st.markdown('''<h6 style='color:grey;font-size:18px'>Copper has been used as construction material for centuries. 
                        It develops a characteristic beautiful green patina, or verdigris, that was highly desired in certain architectural styles, 
                        and still is to this day. Copper is still used today in architecture due to its corrosion resistance, easy workability, 
                        and attractiveness; copper sheets make a beautiful roofing material and other exterior features on buildings.
                        On the interior, copper is used in door handles, trim, vents, railings, kitchen appliances and cookware, 
                        lighting fixtures, and more.''',unsafe_allow_html=True)
            
    with st.container():
        with st.expander(':blue[***Use of Copper in Transportation***]'):
            st.markdown('''<h6 style='color:grey;font-size:18px'>Aside from the copper wiring used in the electrical components of modern cars, copper 
                        and brass have been the industry standard for oil coolers and radiators since the 1970s. Alloys that include copper are used 
                        in the locomotive and aerospace industries as well. As demand for electric cars and other forms of transportation increases,
                        demand for copper components also increases.''',unsafe_allow_html=True)
            
    with st.container():
        with st.expander(':blue[***Other Copper Uses***]'):
            st.markdown('''<h6 style='color:grey;font-size:18px'>Because copper is a beautiful, easily worked material, it is used in art such as copper
                        sheet metal sculptures, jewelry, signage, musical instruments, cookware, and more. The Statue of Liberty, is plated with more than
                        80 tons of copper, which gives her the characteristic pale green patina. Due to its antimicrobial properties, copper is also starting 
                        to gain popularity for high-touch items such as faucets, doorknobs, latches, railings, counters, hooks, handles, and other public 
                        surfaces that tend to gather a lot of germs.''',unsafe_allow_html=True)

    st.link_button('More about copper',url='https://en.wikipedia.org/wiki/Copper')

    col1,col2=st.columns(2)
    with col1:
            st.video('https://www.youtube.com/watch?v=gqmkiPPIsUQ&pp=ygUNIGFib3V0IGNvcHBlcg%3D%3D')
    with col2:
            st.video('https://www.youtube.com/watch?v=AgRYHT6WFV0&pp=ygUTIGNvcHBlciBpbiBpbmR1c3RyeQ%3D%3D')

#set up information for the 'Predictions' menu

if selected == 'Predictions':
    title_text = '''<h1 style='font-size: 32px;text-align: center;color:grey;'>Copper Selling Price and Status Prediction</h1>'''
    st.markdown(title_text, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"]) 
            
    with tab1:    
        
        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col3:               
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
            
        if submit_button and flag==0:
            
            import pickle
            with open(r"model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
            
    with tab2: 
    
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
            
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
            
        if csubmit_button and cflag==0:
            import pickle
            with open(r"cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ') 

#set up information for 'About' menu

if selected == "About":
    st.subheader(':blue[Project Title :]')
    st.markdown('<h5> Industrial Copper Modeling',unsafe_allow_html=True)

    st.subheader(':blue[Domain :]')
    st.markdown('<h5> Manufacturing ',unsafe_allow_html=True)

    st.subheader(':blue[Skills & Technologies :]')
    st.markdown('<h5> Python scripting, Data Preprocessing, Machine learning, EDA, Streamlit ',unsafe_allow_html=True)

    st.subheader(':blue[Overview :]')
    st.markdown('''  <h5>Data Preprocessing:  <br>     
                <li>Loaded the copper CSV into a DataFrame. <br>              
                <li>Cleaned and filled missing values, addressed outliers, and adjusted data types.  <br>           
                <li>Analyzed data distribution and treated skewness.''',unsafe_allow_html=True)
    st.markdown(''' <h5>Feature Engineering: <br>
                <li>Assessed feature correlation to identify potential multicollinearity ''',unsafe_allow_html=True)
    st.markdown('''<h5>Modeling: <br>
                <li >Built a regression model for selling price prediction.
                <li>Built a classification model for status prediction.
                <li>Encoded categorical features and optimized hyperparameters.
                <li>Pickled the trained models for deployment.''',unsafe_allow_html=True)
    st.markdown('''<h5>Streamlit Application: <br>
                <li>Developed a user interface for interacting with the models.
                <li>Predicted selling price and status based on user input.''',unsafe_allow_html=True)
    st.subheader(':blue[About :]')
    st.markdown('''**Hello! I'm Nivesh N P, Having a keen interest in Data science.**''')
    st.link_button('Linkedin','https://www.linkedin.com/in/nivesh-n-p-6b5362186/')
    st.link_button('Github','https://github.com/beingnivesh8')
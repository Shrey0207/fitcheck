import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

# Set the app configuration
st.set_page_config(page_title="FitCheck", page_icon="🔥", layout="wide", initial_sidebar_state='expanded')

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Define the Person class for BMI, BMR, and calorie calculations
class Person:
    def __init__(self, age, height, weight, gender, activity):
        self.age = age
        self.height = height
        self.weight = weight
        self.gender = gender
        self.activity = activity
        self.height_m = height / 100  # Convert height from cm to meters
        self.bmi_calculated = self.weight / (self.height_m ** 2)

    def calculate_bmi(self):
        bmi = round(self.weight / ((self.height / 100) ** 2), 2)
        return bmi

    def display_result(self):
        bmi = self.calculate_bmi()
        bmi_string = f'{bmi} kg/m²'
        if bmi < 18.5:
            category = 'Underweight'
            color = 'Red'
        elif 18.5 <= bmi < 25:
            category = 'Normal'
            color = 'Green'
        elif 25 <= bmi < 30:
            category = 'Overweight'
            color = 'Yellow'
        else:
            category = 'Obesity'
            color = 'Red'
        return bmi_string, category, color


# Configure Google Generative AI
API_KEY = "AIzaSyAo9yfpvJACfzgxPyX3cj3FkSoV4wUy3nY"
genai.configure(api_key=API_KEY)

# Navigation Tabs
st.markdown(
    """
    <style>
        @media (max-width: 600px) {
            .main .block-container {
                padding: 0 1rem;
            }
            .stButton button {
                display: block;
                width: 100%;
            }
        }
        .nav-tabs {
            display: flex;
            flex-wrap: nowrap;
            justify-content: space-around;
            border-bottom: 1px solid #dee2e6;
            margin-bottom: 20px;
        }
        .nav-tabs a {
            display: inline-block;
            padding: 10px 15px;
            margin-right: 2px;
            line-height: 1.5;
            border: 1px solid transparent;
            border-radius: 3px;
            color: #007bff;
            text-decoration: none;
        }
        .nav-tabs a.active {
            color: #495057;
            background-color: #fff;
            border-color: #dee2e6 #dee2e6 #fff;
        }
        .nav-tabs a:hover {
            border-color: #e9ecef #e9ecef #dee2e6;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-around;
        }
        .metric-container > div {
            flex: 1 1 calc(25% - 10px);
            min-width: 150px;
        }
        .stMetric .stMetric-value {
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True
)

tabs = ["Home", "Obesity Prediction", "Calories Calculator", "About Us"]
page = st.sidebar.radio("Navigation", tabs)

# Home Page
if page == "Home":
    st.title("Welcome to FitCheck")
    st.write("Home page content goes here...")

# Obesity Prediction Page
elif page == "Obesity Prediction":
    st.title("Obesity and Waist Size Risk Score Prediction")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    waist = st.number_input("Waist (cm)", min_value=30, max_value=200, value=80)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)

    # Zone selection
    zone = st.selectbox("Zone", options=["N", "S", "E", "W", "NW", "NE", "SW", "SE"])

    # Gender selection
    gender = st.selectbox("Gender", options=["Male", "Female"])

    # Physical activity inputs
    moderate_activity = st.slider("Moderate Physical Activity (level)", min_value=0, max_value=6, value=1)
    vigorous_activity = st.slider("Vigorous Physical Activity (level)", min_value=0, max_value=6, value=1)
    daily_activity = st.slider("Daily Physical Activity (level)", min_value=0, max_value=6, value=1)

    # Predict button
    if st.button("Predict"):
        # Calculate BMI
        person = Person(age, height, weight, gender, 'Moderate exercise (3-5 days/wk)')  # For calorie calculations
        height_m = height / 100
        bmi_calculated = weight / (height_m ** 2)

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Waist': [waist],
            'Weight': [weight],
            'Height': [height],
            'Zone': [zone],
            'Gender': [gender],
            'Moderatephysicalactivity': [moderate_activity],
            'Vigorousphysicalactivity': [vigorous_activity],
            'Dailyphysicalactivity': [daily_activity],
            'Bmi_calculated': [bmi_calculated]
        })

        # Perform the prediction
        prediction = model.predict(input_data)
        risk_score = prediction[0]

        # Determine risk category and color
        if risk_score == 0:
            risk_category = 'Insufficient weight'
            risk_color = 'Blue'
        elif risk_score == 1:
            risk_category = 'Normal weight'
            risk_color = 'Green'
        elif risk_score == 2:
            risk_category = 'Overweight'
            risk_color = 'Yellow'
        elif risk_score == 3:
            risk_category = 'Obesity Type 1'
            risk_color = 'Orange'
        elif risk_score == 4:
            risk_category = 'Obesity Type 2'
            risk_color = 'Blue'
        else:
            risk_category = 'Obesity Type 3'
            risk_color = 'Red'

        # Display the prediction result
        st.write(f"The predicted Obesity and Waist Size Risk Score is: {risk_score}")
        new_title = f'<p style="font-family:sans-serif; color:{risk_color}; font-size: 25px;">{risk_category}</p>'
        st.markdown(new_title, unsafe_allow_html=True)

        # Display BMI result
        st.header('BMI Calculator')
        bmi_string, bmi_category, bmi_color = person.display_result()
        st.metric(label="Body Mass Index (BMI)", value=bmi_string)
        new_title = f'<p style="font-family:sans-serif; color:{bmi_color}; font-size: 25px;">{bmi_category}</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown("Healthy BMI range: 18.5 kg/m² - 25 kg/m².")

# Calories Calculator Page
elif page == "Calories Calculator":
    st.header("Calories Calculator")
    uploaded_file = st.file_uploader("Upload an Image file", accept_multiple_files=False, type=['jpg', 'png', 'jfif'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        bytes_data = uploaded_file.getvalue()
        
        prompt=""" You are a nutritionist and given an uploaded image of a meal, calculate the calories for each individual food item present 
             and provide the results in separate lines. Additionally, include a line for the total calorie count 
             of the entire meal. Please specify any key factors affecting the calculation, such as portion size 
             or specific ingredients visible in the image. Ensure the calorie estimates are as accurate as 
             possible based on the visual information provided.
             Results should should be in the format 
             1. Item1- number of calories
                ----
                ----
             and so on"""


        generate = st.button("Calculate")
        if generate:
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(
                    glm.Content(
                        parts=[
                            glm.Part(text=prompt),
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type='image/jpeg',
                                    data=bytes_data
                                )
                            ),
                        ],
                    ),
                    stream=True
                )

                response.resolve()
                st.write(response.text)
            except:
                st.write("Error! Check the prompt or uploaded image")

# About Us Page
elif page == "About Us":
    st.title("About Us")
    st.write("About Us page content goes here...")

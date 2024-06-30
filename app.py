import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import base64

load_dotenv() ## load all the environment variables

# Set the app configuration
st.set_page_config(page_title="FitCheckr", page_icon="🔥", layout="wide", initial_sidebar_state='expanded')

# Load the trained model
model = joblib.load('decision_tree_model.pkl')


##
def make_image_round(image_path, size):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize((size, size), Image.LANCZOS)

    # Create a mask to make the image round
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Apply the mask to the image
    round_image = Image.new("RGBA", (size, size))
    round_image.paste(image, (0, 0), mask=mask)
    
    return round_image

##
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Local image paths
professor_image_path = "./assets/prof.jpeg"
team1_image_path = "./assets/team1.jpg"
team2_image_path = "./assets/team2.jpg"
team3_image_path = "./assets/team3.jpg"


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

    def calculate_bmr(self):
        if self.gender == 'Male':
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        return bmr

    def calories_calculator(self):
        activities = ['Little/no exercise', 'Light exercise', 'Moderate exercise (3-5 days/wk)', 'Very active (6-7 days/wk)', 'Extra active (very active & physical job)']
        weights = [1.2, 1.375, 1.55, 1.725, 1.9]
        weight_factor = weights[activities.index(self.activity)]
        maintain_calories = self.calculate_bmr() * weight_factor
        return maintain_calories

# Configure Google Generative AI
genai.configure(api_key=os.getenv("API_KEY"))

# Navigation Bar
page = st.sidebar.selectbox("FitCheck", ["Home", "Obesity Prediction", "Calorie Calculator", "About Us"])

# Home Page
if page == "Home":
    st.title("Welcome to FitCheck")
    st.write("FitCheck is designed to help you monitor and improve your health. Here are the key features:")

    # Feature 1: Machine Learning Model
    st.header("1. Weight Level Prediction")
    st.markdown("""
    FitCheck uses a decision tree algorithm to predict the weight level of a person based on factors such as weight, height, gender, age, waist size, and geographical location. This model emphasizes the impact of abdominal obesity, a prevalent issue in India.
    """)

    # Feature 2: Calorie Recommendation
    st.header("2. Calorie Recommendation")
    st.markdown("""
    FitCheck advises users on their calorie intake based on their desired weight loss goals:
    
    - **Maintain Weight:** Calories required to maintain current weight.
    - **Mild Weight Loss (0.25 kg/week):** Calories adjusted for mild weight loss goals.
    - **Weight Loss (0.5 kg/week):** Calories adjusted for moderate weight loss goals.
    - **Extreme Weight Loss (1 kg/week):** Calories adjusted for aggressive weight loss goals.
    
    This calculation considers the user's basal metabolic rate (BMR) and physical activity level categorized as:
    
    - **Little/No Exercise**
    - **Light Exercise**
    - **Moderate Exercise (3-5 days/week)**
    - **Very Active (6-7 days/week)**
    - **Extra Active (Physically demanding job)**
    """)

    # Feature 3: Calorie Calculator App
    st.header("3. Calorie Calculator")
    st.markdown("""
    FitCheck includes a calorie calculator app powered by Google's Gemini Pro Vision model. This tool analyzes food images to accurately calculate total calories, helping users track their calorie intake effectively.
    """)

    # Additional content or sections can be added as needed
    
    # Adding style for circular images and Font Awesome icons (if applicable)
    st.markdown("""
    <style>
        /* Add additional CSS styles specific to your content here */
    </style>
    """, unsafe_allow_html=True)

    # End of main content


# Obesity Prediction Page
elif page == "Obesity Prediction":
    st.title("Obesity and Waist Size Risk Score Prediction")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    waist = st.number_input("Waist (cm)", min_value=30, max_value=200, value=80)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)

    # Zone selection
    zone = st.selectbox("Zone", options=["N", "S", "E", "W", "NW", "NE"])

    # Gender selection
    gender = st.selectbox("Gender", options=["Male", "Female"])
    activity = st.select_slider('Activity',options=['Little/no exercise', 'Light exercise', 'Moderate exercise (3-5 days/wk)', 'Very active (6-7 days/wk)', 
    'Extra active (very active & physical job)'])

    # Predict button
    if st.button("Predict"):
        # Calculate BMI
        person = Person(age, height, weight, gender, activity)  # For calorie calculations
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
            risk_color = 'Red'
        else:
            risk_category = 'Obesity Type 3'
            risk_color = 'Dark Red'

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

        # Display calories calculator result
        st.header('Calories Calculator')
        maintain_calories = person.calories_calculator()
        st.write('The results show a number of daily calorie estimates that can be used as a guideline for how many calories to consume each day to maintain, lose, or gain weight at a chosen rate.')
        plans = ["Maintain weight", "Mild weight loss", "Weight loss", "Extreme weight loss"]
        weights = [1, 0.9, 0.8, 0.6]
        losses = ['-0 kg/week', '-0.25 kg/week', '-0.5 kg/week', '-1 kg/week']
        for plan, weight_factor, loss, col in zip(plans, weights, losses, st.columns(4)):
            with col:
                st.metric(label=plan, value=f'{round(maintain_calories * weight_factor)} Calories/day', delta=loss, delta_color="inverse")

# Calories Calculator Page
elif page == "Calorie Calculator":
    st.title("Gemini NutriAI 🍽️")
    # Sidebar guide
    st.sidebar.markdown("""
    ### Guide
    1. Enter your Gemini API key in the provided input field.
    2. The default prompt is already integrated into the application.
    3. Simply upload a photo of food to receive calorie information.
    4. For additional details or a custom prompt, utilize the "Provide Prompt" input to access extra information via the Gemini Vision Pro model.
    """)
    # Configure Google Gemini Pro Vision API with the API key from the input field
    api_key = st.sidebar.text_input("Enter your Google API Key:", key="api_key")
    if api_key:
         genai.configure(api_key=api_key)
    else:
        st.warning("Please enter your Google API Key.")

    # Guide for obtaining Google API Key if not available
    st.sidebar.subheader("Don't have a Google API Key?")
    st.sidebar.write("Visit [Google Makersuite](https://makersuite.google.com/app/apikey) and log in with your Google account. Then click on 'Create API Key'.")
    ## Function to load Google Gemini Pro Vision API And get response
    def get_gemini_response(input, image, prompt):
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([input, image[0], prompt])
        return response.text
    def input_image_setup(uploaded_file):
        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()

            image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
        
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Tell me the total calories")

    input_prompt=""" You are a nutritionist and given an uploaded image of a meal, calculate the calories for each individual food item present 
             and provide the results in separate lines. Additionally, include a line for the total calorie count 
             of the entire meal. Please specify any key factors affecting the calculation, such as portion size 
             or specific ingredients visible in the image. Ensure the calorie estimates are as accurate as 
             possible based on the visual information provided.
             Results should should be in the format 
             1. Item1- number of calories
                ----
                ----
             and so on"""
    
    ## If submit button is clicked
    if submit:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, "")
        st.subheader("The Response is")
        st.write(response)

else:
    st.title("About Us")
        # Professor Section
    st.header("Project Guide")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        professor_image = make_image_round(professor_image_path, 150)
        professor_image_base64 = get_image_base64(professor_image)
        st.markdown(f"""
        <div class="card">
            <img src="data:image/png;base64,{professor_image_base64}">
            <div class="container">
                <b>Dr Vinpin Chandra Pal</b><br>
                <a href="mailto:vipin@ei.nits.ac.in"><i class="fa fa-envelope"></i></a>
                <a href="http://eie.nits.ac.in/vipin/"><i class="fa fa-globe"></i></a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Team Section
    st.header("Our Team")
    team_members = [
        {"name": "Shrey Tolasaria", "image_path": team1_image_path, "email": "shreytolasaria4297@gmail.com", "linkedin": "https://www.linkedin.com/in/shrey-tolasaria-176381231/", "github": "https://github.com/Shrey0207"},
        {"name": "Kaushik Borah", "image_path": team2_image_path, "email": "kaushikborah4080@gmail.com", "linkedin": "https://www.linkedin.com/in/kaushik-borah-317758226/", "github": "https://github.com/dngeonMaster1706"},
        {"name": "Hritik Baranwal ", "image_path": team3_image_path, "email": "hritik21_ug@ei.nits.ac.in", "linkedin": "https://www.linkedin.com/in/hritik-baranwal-b65729237/", "github": "https://github.com/hritik06"}
    ]

    cols = st.columns(3)

    for idx, member in enumerate(team_members):
        with cols[idx]:
            member_image = make_image_round(member["image_path"], 100)
            member_image_base64 = get_image_base64(member_image)
            st.markdown(f"""
            <div class="card">
                <img src="data:image/png;base64,{member_image_base64}">
                <div class="container">
                    <b>{member['name']}</b><br>
                    <a href="mailto:{member['email']}"><i class="fa fa-envelope"></i></a>
                    <a href="{member['linkedin']}"><i class="fa fa-linkedin"></i></a>
                    <a href="{member['github']}"><i class="fa fa-github"></i></a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Adding style for circular images and Font Awesome icons
st.markdown("""
<style>
    .card {
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        width: 80%;
        border-radius: 10px;
        text-align: center;
        margin: 10px auto;
        padding: 10px;
    }
    .card img {
        border-radius: 50%;
        width: 50%;
        margin: 10px 0;
    }
    .container {
        padding: 2px 16px;
    }
    .fa {
        font-size: 20px;
        margin: 0 10px;
    }
    .fa:hover {
        color: #1e90ff;
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""", unsafe_allow_html=True)
    
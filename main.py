from fastapi import FastAPI # used to creating api
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # used to setup the data that will posted in api
import pickle # loading the save model
import json # javascript object notation (used to store data in structured manner like dictionary)


app = FastAPI()
origin=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credential="True",
    allow_methods=['*'],
    allow_head=["*"]
)

class model_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       
        
# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

@app.post('/diabetes_prediction') # append this  in url section
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data) # convert json to dictionary
    
    preg = input_dictionary['pregnancies']  # key gives values
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list]) # instead of reshape we provide list of list
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    

    #uvicorn diabetes_ml_api:app

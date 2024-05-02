import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from enum import Enum
import uvicorn
import pickle


# Define enum for race choices
class Race(str, Enum):
    white = "White"
    black = "Black"
    multiracial = "Multiracial"
    hispanic = "Hispanic"
    other = "Other"


# Define enum for age ranges
class Age(str, Enum):
    age_18_to_24 = "18 to 24"
    age_25_to_29 = "25 to 29"
    age_30_to_34 = "30 to 34"
    age_35_to_39 = "35 to 39"
    age_40_to_44 = "40 to 44"
    age_45_to_49 = "45 to 49"
    age_50_to_54 = "50 to 54"
    age_55_to_59 = "55 to 59"
    age_60_to_64 = "60 to 64"
    age_65_to_69 = "65 to 69"
    age_70_to_74 = "70 to 74"
    age_75_to_79 = "75 to 79"
    age_80_or_older = "80 or older"


# Define enum for sex choices
class Sex(str, Enum):
    male = "Male"
    female = "Female"


# Define enum for general health choices
class GeneralHealth(str, Enum):
    excellent = "Excellent"
    very_good = "Very Good"
    good = "Good"
    fair = "Fair"
    poor = "Poor"


# Define enum for yes/no choices
class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


# Define enum for smoker status choices
class SmokerStatus(str, Enum):
    current = "Current"
    former = "Former"
    never = "Never"


API_KEY = "lou"

# Create a FastAPI app instance
app = FastAPI()


# Define a dependency to check the API key
async def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# Define prediction endpoint
@app.get("/predict")
async def predict(
    api_key_verified: bool = Depends(verify_api_key),
    race: Race = Query(description="Race"),
    age: Age = Query(description="Age group"),
    height: float = Query(description="Height in meters"),
    sex: Sex = Query(description="Sex"),
    general_health: GeneralHealth = Query(description="Assessment of general health"),
    physical_activities: YesNo = Query(description="Engagement in physical activities"),
    difficulty_walking: YesNo = Query(description="Difficulty walking"),
    smoker_status: SmokerStatus = Query(description="Smoking status"),
    alcohol_drinkers: YesNo = Query(description="Alcohol consumption"),
    physical_health_days: float = Query(description="Number of days with physical health issues"),
    mental_health_days: float = Query(description="Number of days with mental health issues"),
    sleep_hours: float = Query(description="Number of hours of sleep per day"),
    bmi: float = Query(description="Body mass index (BMI)"),
    weight: float = Query(description="Weight in kilograms")
):
    try:
        # Convert inputs to DataFrame
        data = pd.DataFrame({
            "Race": [race.value],
            "Age": [age.value],
            "Height (m)": [height],
            "Sex": [sex.value],
            "GeneralHealth": [general_health.value],
            "PhysicalActivities": [physical_activities.value],
            "DifficultyWalking": [difficulty_walking.value],
            "SmokerStatus": [smoker_status.value],
            "AlcoholDrinkers": [alcohol_drinkers.value],
            "PhysicalHealthDays": [physical_health_days],
            "MentalHealthDays": [mental_health_days],
            "SleepHours": [sleep_hours],
            "BMI": [bmi],
            "Weight (kg)": [weight]
        })

        # Load the logistic regression model and constants
        with open('logistic_heart_disease.pkl', 'rb') as file:
            scaler, needed_columns, model = pickle.load(file)

        # Scale the numerical columns
        numerical_columns = pd.DataFrame(data[data.select_dtypes(include=["float", "int"]).columns])
        scaled_numerical_columns = pd.DataFrame(scaler.fit_transform(numerical_columns), columns=numerical_columns.columns)

        # Encode the categorical columns
        categorical_columns = pd.DataFrame(data[data.select_dtypes(include=["object"]).columns])
        encoded_categorical = pd.get_dummies(categorical_columns)

        # Concat the columns together and then add an array of false for missing encoded columns
        data_processed = pd.concat([scaled_numerical_columns, encoded_categorical], axis=1)

        # Get columns present in the list but not in the DataFrame
        columns_to_add = set(needed_columns) - set(data_processed.columns)

        # Add columns to DataFrame with False values
        for column in columns_to_add:
            data_processed[column] = False
        data_processed = data_processed.reindex(columns=needed_columns)
        # Run the Logistic Regression Model
        prediction = model.predict(data_processed)
        if prediction[0] == 0:
            pred_return = "You have a lower probability for heart disease. Good for you!"
        elif prediction[0] == 1:
            pred_return = "Oh no! There is a higher probability that you could contract heart disease."
        # Return prediction result
        return pred_return
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

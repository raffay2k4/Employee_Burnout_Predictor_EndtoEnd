from fastapi import FastAPI,Query
from model_io import load_model
from typing import Literal
import numpy as np

app=FastAPI()

loaded_model=load_model()

@app.get("/")
async def home_page():
    return "This is our home page"

@app.post("/prediction_endpoint")
async def predict(
    Gender: Literal['Male', 'Female'] = Query(...,description="The gender of the employee (Male/Female)") , 
    Company_Type: Literal['Product', 'Service'] = Query(...,description="The type of company where the employee is working (Service/Product)"), 
    WFH_Setup_Available: Literal['Yes', 'No'] = Query(...,description="Is the work from home facility available for the employee (Yes/No)") , 
    Designation: float = Query(...,description="The designation of the employee of work in the organization (In the range of 1-5 where bigger is higher designation.)") , 
    Resource_Allocation: float = Query(...,description="The amount of resource allocated to the employee to work, ie. number of working hours. (In the range of 1-10 where higher means more resource)") , 
    Mental_Fatigue_Score: float = Query(...,description="The level of fatigue mentally the employee is facing. (In the range of [0.0, 10.0] where 0.0 means no fatigue and 10.0 means completely fatigue)") ,
    Days_at_Company: float = Query(...,description="The number of days since employee joined their current company.")
):

    model_inputs=[]
    
    if Gender=="Male":
        model_inputs.append(int(1))  
    elif Gender=="Female":
        model_inputs.append(int(0))
        

    if Company_Type=="Product":
        model_inputs.append(int(0))  
    elif Company_Type=="Service":
        model_inputs.append(int(1))


    if WFH_Setup_Available=="Yes":
        model_inputs.append(int(1))  
    elif WFH_Setup_Available=="No":
        model_inputs.append(int(0))


    model_inputs.append(float(round(Designation)))
    model_inputs.append(float(round(Resource_Allocation)))
    model_inputs.append(float(Mental_Fatigue_Score))
    model_inputs.append(float(round(Days_at_Company)))


    model_inputs = np.array(model_inputs).reshape(1, -1) 

    predicted_burnout = loaded_model.predict(model_inputs)

    return {f"predicted_burnout: {predicted_burnout[0]}"}  
    
    
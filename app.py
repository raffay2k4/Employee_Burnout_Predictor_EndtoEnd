import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/prediction_endpoint"

def predict(gender, company_type, wfh_setup, designation, resource_allocation, mental_fatigue_score, days_at_company):
    params = {
        "Gender": gender,
        "Company_Type": company_type,
        "WFH_Setup_Available": wfh_setup,
        "Designation": designation,
        "Resource_Allocation": resource_allocation,
        "Mental_Fatigue_Score": mental_fatigue_score,
        "Days_at_Company": days_at_company
    }
    

    response = requests.post(API_URL, params=params)
    
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}, {response.text}"


inputs = [
    gr.Dropdown(["Male", "Female"], label="The gender of the employee (Male/Female)"),
    gr.Dropdown(["Product", "Service"], label="The type of company where the employee is working (Service/Product)"),
    gr.Dropdown(["Yes", "No"], label="Is the work from home facility available for the employee (Yes/No)"),
    gr.Number(minimum=1, maximum=5, step=1, label="The designation of the employee of work in the organization (In the range of 1-5 where bigger is higher designation.)"),
    gr.Number(minimum=1, maximum=10, step=1, label="The amount of resource allocated to the employee to work, ie. number of working hours. (In the range of 1-10 where higher means more resource)"),
    gr.Number(minimum=1, maximum=10, step=0.1, label="The level of fatigue mentally the employee is facing. (In the range of [0.0, 10.0] where 0.0 means no fatigue and 10.0 means completely fatigue)"),
    gr.Number(minimum=1, step=1000, label="The number of days since employee joined their current company.")
]

outputs = gr.Textbox(label="Predicted Burnout")


gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Employee Burnout Prediction").launch(debug=True)


#To deploy on HuggingFace space:

#Create a new space with SDK set as gradio
#Upload all the files on space
#Run api.py script to activate our API
#Expose the port of API to ngrok to host API on the internet
#Edit app.py file on HuggingFace space and set API_URL to the URL provided by ngrok.
#Build App
import joblib,pydantic
from fastapi import FastAPI

app = FastAPI()
model = joblib.load('iris_model.pkl')

class InputData(pydantic.BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict')
def predict(data: InputData):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(input_data)
    return {'prediction': int(prediction[0])}

@app.get('/health')
def health():
    return {'status': 'ok'}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.http as httpclient
import pymongo
import numpy as np
import datetime
import logging

# Define the request model
class InferenceRequest(BaseModel):
    region: str
	 vintage: int

# Initialize FastAPI app
app = FastAPI()

# Triton server details
url = "localhost:8000"
model_name = "sample_lstm_model"

# MongoDB details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DATABASE_NAME = "neuroflow"
COLLECTION_NAME = "result_log" 

# Initialize Triton client
client = httpclient.InferenceServerClient(url)

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

@app.post("/infer/")
async def infer(data: InferenceRequest):

	  wine_data = np.random.rand(1, 77).astype(np.float32)  # TODO: 몽고에서 불러오기!
    climate_data = np.random.rand(1, 7, 22).astype(np.float32)  # TODO: 몽고에서 불러오기!

    # Create the input tensors
    inputs = [
        httpclient.InferInput("wine_data__0", wine_data.shape, "FP32"),
        httpclient.InferInput("climate_data__1", climate_data.shape, "FP32")
    ]

    # Set the data for the inputs
    inputs[0].set_data_from_numpy(wine_data)
    inputs[1].set_data_from_numpy(climate_data)

    # Send request to Triton and get response
    try:
        response = client.infer(model_name, inputs)
        result = float(response.as_numpy('output__0')[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save data and result to MongoDB
    record = {
        "timestamp": datetime.datetime.now(),
        "request": data.dict(),
        "model_input": {
            "wine_data": wine_data.tolist(),
            "climate_data": climate_data.tolist()
        },
        "model_output": result
    }
    try:
        collection.insert_one(record)
    except Exception as e:
        logging.error(f"Error saving to MongoDB: {e}")

    # TODO : 여기서 infer 결과랑 실제 값(Data-baseModel) 비교해서 return

    return {"input": data.dict(), "inference_result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

import os
import tempfile
import shutil
from typing import List
import random
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load the plant disease dataset
def load_dataset():
    try:
        df = pd.read_csv("assets/plant_disease_data.csv")
        return df
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        return None

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="A simple API for plant disease detection using image uploads",
    version="1.0.0"
)

# CORS settings to allow local frontend
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset at startup
dataset = load_dataset()

# Pydantic models
class PredictionResponse(BaseModel):
    infection: str
    confidence: float
    plant_type: str
    additional_info: dict

class HealthResponse(BaseModel):
    status: str
    message: str

# Disease types based on the dataset
DISEASE_TYPES = [
    "Healthy",
    "Mild Infection", 
    "Severe Infection"
]

# Plant types from the dataset
PLANT_TYPES = [
    "Corn", "Potato", "Rice", "Wheat", "Tomato"
]

# Common disease names for different plant types
PLANT_DISEASES = {
    "Corn": ["Corn Leaf Blight", "Corn Rust", "Corn Smut"],
    "Potato": ["Late Blight", "Early Blight", "Blackleg"],
    "Rice": ["Rice Blast", "Bacterial Leaf Blight", "Sheath Blight"],
    "Wheat": ["Wheat Rust", "Powdery Mildew", "Fusarium Head Blight"],
    "Tomato": ["Early Blight", "Late Blight", "Septoria Leaf Spot"]
}

def generate_dummy_prediction() -> dict:
    """Generate a realistic dummy prediction based on the dataset"""
    if dataset is not None:
        # Get a random sample from the dataset
        sample = dataset.sample(1).iloc[0]
        plant_type = sample['Plant_Type']
        disease_status = sample['Disease_Status']
        
        # Map disease status to specific disease names
        if disease_status == "Healthy":
            infection = "Healthy"
            confidence = random.uniform(0.85, 0.98)
        elif disease_status == "Mild Infection":
            diseases = PLANT_DISEASES.get(plant_type, ["Leaf Spot Disease"])
            infection = random.choice(diseases)
            confidence = random.uniform(0.70, 0.89)
        else:  # Severe Infection
            diseases = PLANT_DISEASES.get(plant_type, ["Severe Leaf Disease"])
            infection = random.choice(diseases)
            confidence = random.uniform(0.60, 0.85)
            
        additional_info = {
            "humidity": round(sample['Humidity'], 2),
            "temperature": round(sample['Temperature'], 2),
            "leaf_color": sample['Leaf_Color'],
            "leaf_spot_size": round(sample['Leaf_Spot_Size'], 2)
        }
    else:
        # Fallback if dataset can't be loaded
        plant_type = random.choice(PLANT_TYPES)
        infection = random.choice(DISEASE_TYPES)
        confidence = random.uniform(0.60, 0.95)
        additional_info = {
            "humidity": round(random.uniform(30, 90), 2),
            "temperature": round(random.uniform(15, 35), 2),
            "leaf_color": random.choice(["Green", "Yellow", "Brown"]),
            "leaf_spot_size": round(random.uniform(0, 10), 2)
        }
    
    return {
        "infection": infection,
        "confidence": round(confidence, 3),
        "plant_type": plant_type,
        "additional_info": additional_info
    }

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Plant Disease Detection API is running"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(image: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        image: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with disease prediction and confidence
    """
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    # Validate file size (max 10MB)
    if image.size and image.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail="File size must be less than 10MB"
        )
    
    # Save image temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, image.filename or "uploaded_image")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # For now, generate dummy prediction
        prediction = generate_dummy_prediction()
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@app.get("/dataset-info")
async def get_dataset_info():
    """Get information about the loaded dataset"""
    if dataset is None:
        return {"error": "Dataset not loaded"}
    
    return {
        "total_records": len(dataset),
        "plant_types": dataset['Plant_Type'].unique().tolist(),
        "disease_statuses": dataset['Disease_Status'].unique().tolist(),
        "columns": dataset.columns.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

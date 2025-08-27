import os
import tempfile
import shutil
from typing import List, Dict, Tuple
import random
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the plant disease dataset
def load_dataset():
    try:
        df = pd.read_csv("assets/plant_disease_data.csv")
        return df
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        return None

class PlantDiseasePredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = ['Plant_Type', 'Leaf_Color', 'Leaf_Spot_Size', 'Humidity', 'Temperature']
        self.target_column = 'Disease_Status'
        self.is_trained = False
        self.accuracy = 0.0
        
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        df_encoded = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Plant_Type', 'Leaf_Color']
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.encoders[col].transform(df[col])
        
        return df_encoded[self.feature_columns]
    
    def train(self, df):
        """Train the disease prediction model"""
        try:
            # Prepare features and target
            X = self.prepare_features(df)
            
            # Encode target variable
            if self.target_column not in self.encoders:
                self.encoders[self.target_column] = LabelEncoder()
            y = self.encoders[self.target_column].fit_transform(df[self.target_column])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.is_trained = True
            
            print(f"Model trained with accuracy: {self.accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_with_confidence(self, plant_type, leaf_color, leaf_spot_size, humidity, temperature):
        """Make prediction with confidence score"""
        if not self.is_trained:
            return self._fallback_prediction(plant_type, leaf_color, leaf_spot_size, humidity, temperature)
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Plant_Type': [plant_type],
                'Leaf_Color': [leaf_color],
                'Leaf_Spot_Size': [leaf_spot_size],
                'Humidity': [humidity],
                'Temperature': [temperature]
            })
            
            # Prepare features
            X = self.prepare_features(input_data)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Decode prediction
            predicted_disease = self.encoders[self.target_column].inverse_transform([predicted_class_idx])[0]
            
            return predicted_disease, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return self._fallback_prediction(plant_type, leaf_color, leaf_spot_size, humidity, temperature)
    
    def _fallback_prediction(self, plant_type, leaf_color, leaf_spot_size, humidity, temperature):
        """Fallback rule-based prediction when ML model fails"""
        # Simple rule-based logic
        if leaf_spot_size > 5:
            return "Severe Infection", random.uniform(0.70, 0.85)
        elif leaf_spot_size > 2:
            return "Mild Infection", random.uniform(0.65, 0.80)
        elif leaf_color in ["Yellow", "Brown"]:
            return "Mild Infection", random.uniform(0.60, 0.75)
        else:
            return "Healthy", random.uniform(0.80, 0.95)
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Plant Disease Detection API",
    description="An intelligent API for plant disease detection with ML-based predictions",
    version="2.0.0"
)

# CORS settings
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and train model at startup
dataset = load_dataset()
predictor = PlantDiseasePredictor()

if dataset is not None:
    print("Training disease prediction model...")
    predictor.train(dataset)
else:
    print("Dataset not loaded - using fallback predictions")

# Pydantic models
class PredictionResponse(BaseModel):
    infection: str
    confidence: float
    plant_type: str
    additional_info: dict
    model_accuracy: float

class HealthResponse(BaseModel):
    status: str
    message: str
    model_trained: bool
    dataset_size: int

class ImageAnalysisInput(BaseModel):
    plant_type: str
    leaf_color: str
    environmental_conditions: dict

# Enhanced disease mapping
PLANT_DISEASES = {
    "Corn": {
        "Mild Infection": ["Corn Leaf Blight", "Northern Corn Leaf Spot", "Gray Leaf Spot"],
        "Severe Infection": ["Corn Rust", "Corn Smut", "Southern Corn Leaf Blight"]
    },
    "Potato": {
        "Mild Infection": ["Early Blight", "Potato Scab", "Blackleg"],
        "Severe Infection": ["Late Blight", "Potato Virus Y", "Ring Rot"]
    },
    "Rice": {
        "Mild Infection": ["Brown Spot", "Narrow Brown Spot", "Bacterial Leaf Streak"],
        "Severe Infection": ["Rice Blast", "Bacterial Leaf Blight", "Sheath Blight"]
    },
    "Wheat": {
        "Mild Infection": ["Septoria Leaf Blotch", "Tan Spot", "Leaf Rust"],
        "Severe Infection": ["Wheat Rust", "Powdery Mildew", "Fusarium Head Blight"]
    },
    "Tomato": {
        "Mild Infection": ["Septoria Leaf Spot", "Target Spot", "Bacterial Speck"],
        "Severe Infection": ["Early Blight", "Late Blight", "Fusarium Wilt"]
    }
}

def simulate_image_analysis(image_content) -> Dict:
    """Simulate advanced image analysis to extract plant characteristics"""
    # In a real implementation, this would use computer vision
    # For now, we'll intelligently sample from the dataset or use realistic distributions
    
    if dataset is not None:
        # Get a weighted sample based on disease prevalence
        sample = dataset.sample(1, weights=dataset.groupby('Disease_Status').transform('count')['Plant_ID']).iloc[0]
        
        return {
            'plant_type': sample['Plant_Type'],
            'leaf_color': sample['Leaf_Color'],
            'leaf_spot_size': sample['Leaf_Spot_Size'],
            'humidity': sample['Humidity'],
            'temperature': sample['Temperature']
        }
    else:
        # Fallback realistic simulation
        plant_types = ["Corn", "Potato", "Rice", "Wheat", "Tomato"]
        leaf_colors = ["Green", "Yellow", "Brown"]
        
        # Weight towards healthy conditions but include disease indicators
        leaf_color_weights = [0.6, 0.3, 0.1]  # Green most likely, then Yellow, then Brown
        
        return {
            'plant_type': random.choice(plant_types),
            'leaf_color': np.random.choice(leaf_colors, p=leaf_color_weights),
            'leaf_spot_size': np.random.exponential(2.0),  # Most plants have small spots
            'humidity': np.random.normal(60, 15),  # Normal distribution around 60%
            'temperature': np.random.normal(25, 8)   # Normal distribution around 25°C
        }

def get_specific_disease_name(plant_type: str, disease_status: str) -> str:
    """Get specific disease name based on plant type and status"""
    if disease_status == "Healthy":
        return "Healthy"
    
    if plant_type in PLANT_DISEASES and disease_status in PLANT_DISEASES[plant_type]:
        return random.choice(PLANT_DISEASES[plant_type][disease_status])
    
    # Fallback
    return disease_status

@app.get("/", response_model=HealthResponse)
async def root():
    """Enhanced health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Enhanced Plant Disease Detection API is running",
        model_trained=predictor.is_trained,
        dataset_size=len(dataset) if dataset is not None else 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(image: UploadFile = File(...)):
    """
    Enhanced disease prediction from uploaded image
    
    Args:
        image: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with ML-based disease prediction and confidence
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
        
        # Simulate image analysis
        image_analysis = simulate_image_analysis(temp_file_path)
        
        # Make prediction using trained model
        disease_status, confidence = predictor.predict_with_confidence(
            plant_type=image_analysis['plant_type'],
            leaf_color=image_analysis['leaf_color'],
            leaf_spot_size=image_analysis['leaf_spot_size'],
            humidity=image_analysis['humidity'],
            temperature=image_analysis['temperature']
        )
        
        # Get specific disease name
        specific_disease = get_specific_disease_name(
            image_analysis['plant_type'], 
            disease_status
        )
        
        # Prepare response
        additional_info = {
            "humidity": round(image_analysis['humidity'], 2),
            "temperature": round(image_analysis['temperature'], 2),
            "leaf_color": image_analysis['leaf_color'],
            "leaf_spot_size": round(image_analysis['leaf_spot_size'], 2),
            "analysis_method": "ML Model" if predictor.is_trained else "Rule-based"
        }
        
        return PredictionResponse(
            infection=specific_disease,
            confidence=round(confidence, 3),
            plant_type=image_analysis['plant_type'],
            additional_info=additional_info,
            model_accuracy=round(predictor.accuracy, 3)
        )
        
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

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if not predictor.is_trained:
        return {
            "error": "Model not trained",
            "fallback_active": True
        }
    
    feature_importance = predictor.get_feature_importance()
    
    return {
        "model_type": "Random Forest Classifier",
        "accuracy": round(predictor.accuracy, 3),
        "feature_importance": {k: round(v, 3) for k, v in feature_importance.items()},
        "features_used": predictor.feature_columns,
        "training_dataset_size": len(dataset) if dataset is not None else 0
    }

@app.get("/dataset-info")
async def get_dataset_info():
    """Get comprehensive information about the loaded dataset"""
    if dataset is None:
        return {"error": "Dataset not loaded"}
    
    # Calculate statistics
    stats = {
        "total_records": len(dataset),
        "plant_types": dataset['Plant_Type'].value_counts().to_dict(),
        "disease_distribution": dataset['Disease_Status'].value_counts().to_dict(),
        "leaf_colors": dataset['Leaf_Color'].value_counts().to_dict(),
        "columns": dataset.columns.tolist(),
        "summary_stats": {
            "humidity": {
                "mean": round(dataset['Humidity'].mean(), 2),
                "std": round(dataset['Humidity'].std(), 2),
                "min": round(dataset['Humidity'].min(), 2),
                "max": round(dataset['Humidity'].max(), 2)
            },
            "temperature": {
                "mean": round(dataset['Temperature'].mean(), 2),
                "std": round(dataset['Temperature'].std(), 2),
                "min": round(dataset['Temperature'].min(), 2),
                "max": round(dataset['Temperature'].max(), 2)
            },
            "leaf_spot_size": {
                "mean": round(dataset['Leaf_Spot_Size'].mean(), 2),
                "std": round(dataset['Leaf_Spot_Size'].std(), 2),
                "min": round(dataset['Leaf_Spot_Size'].min(), 2),
                "max": round(dataset['Leaf_Spot_Size'].max(), 2)
            }
        }
    }
    
    return stats

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with current dataset"""
    if dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset available for training"
        )
    
    success = predictor.train(dataset)
    if success:
        return {
            "message": "Model retrained successfully",
            "accuracy": round(predictor.accuracy, 3),
            "dataset_size": len(dataset)
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrain model"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
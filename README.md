# Plant Disease Detection API

A minimal FastAPI backend prototype for plant disease detection using image uploads.

## Features

- **Image Upload**: Accepts image files via multipart/form-data
- **Dummy ML Predictions**: Returns realistic predictions based on the plant disease dataset
- **Dataset Integration**: Uses the provided plant disease dataset for realistic responses
- **API Documentation**: Auto-generated FastAPI docs
- **Health Check**: Basic health monitoring endpoint

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Upload image and get disease prediction
- `GET /dataset-info` - Information about the loaded dataset

## Response Format

The `/predict` endpoint returns:

```json
{
  "infection": "Corn Leaf Blight",
  "confidence": 0.847,
  "plant_type": "Corn",
  "additional_info": {
    "humidity": 69.49,
    "temperature": 30.68,
    "leaf_color": "Brown",
    "leaf_spot_size": 3.8
  }
}
```

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

#### Option 1: Using uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 2: Using Python
```bash
python main.py
```

### 3. Access the API

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/

# Upload image and get prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg"
```

### Using the Interactive Docs

1. Open http://localhost:8000/docs in your browser
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Upload an image file
5. Click "Execute"

## Dataset Information

The API uses the `assets/plant_disease_data.csv` file which contains:
- **Plant Types**: Corn, Potato, Rice, Wheat, Tomato
- **Disease Statuses**: Healthy, Mild Infection, Severe Infection
- **Features**: Leaf Color, Leaf Spot Size, Humidity, Temperature

## Project Structure

```
Plant-Disease-Detection/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── assets/
    └── plant_disease_data.csv  # Plant disease dataset
```

## Notes

- This is a prototype that generates dummy predictions
- In a production environment, you would integrate with an actual ML model
- Images are temporarily saved and then cleaned up
- Maximum file size is limited to 10MB
- Only image files are accepted

## Future Enhancements

- Integrate with actual ML models (TensorFlow, PyTorch)
- Add image preprocessing and validation
- Implement caching for predictions
- Add authentication and rate limiting
- Support for batch image processing 
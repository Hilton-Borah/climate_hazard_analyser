from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from bson import ObjectId
import json
from weather_processor import WeatherProcessor
import pandas as pd
from database import Database
from export_handler import ExportHandler
from statistical_analysis import StatisticalAnalysis
from fastapi.responses import StreamingResponse

# Load environment variables
load_dotenv()

# Initialize WeatherProcessor
weather_processor = WeatherProcessor()

# Custom Pydantic models with proper serialization
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class RegionInput(BaseModel):
    region_type: str  # 'city', 'state', or 'custom'
    name: Optional[str] = None
    coordinates: Optional[List[List[float]]] = None  # For custom regions: [[lat, lon], ...]

    class Config:
        json_encoders = {ObjectId: str}

class AnalysisRequest(BaseModel):
    region: RegionInput
    start_year: int
    end_year: int
    hazard_type: str  # 'heatwave', 'drought', or 'rainfall'

    class Config:
        json_encoders = {ObjectId: str}

class AnalysisResponse(BaseModel):
    id: str
    region: RegionInput
    hazard_type: str
    period: str
    trends: Dict[str, float]
    summary: Dict[str, Any]

    class Config:
        json_encoders = {ObjectId: str}

class HazardTrend(BaseModel):
    frequency: float
    intensity: float
    duration: float
    trend_percentage: float

    class Config:
        json_encoders = {ObjectId: str}

app = FastAPI(
    title="Climate Hazard Trend Analyser API",
    description="API for analyzing and visualizing climate hazard trends",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URL", "")  # Changed from MONGODB_URI to MONGODB_URL
DATABASE_NAME = os.getenv("DATABASE_NAME", "climate_hazards")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DATABASE_NAME] 

# Helper function to convert ObjectId to string
def convert_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {key: convert_objectid(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    return obj

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

@app.get("/")
async def root():
    return {"message": "Climate Hazard Trend Analyser API"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_hazards(request: AnalysisRequest):
    try:
        # print("Received request:", request)
        current_year = datetime.now().year
        if not (1900 <= request.start_year <= current_year and 
                1900 <= request.end_year <= current_year and 
                request.start_year <= request.end_year):
            raise HTTPException(status_code=400, detail="Invalid year range")

        try:
            # Get coordinates based on region type
            if request.region.region_type == 'city':
                if not request.region.name:
                    raise HTTPException(status_code=400, detail="City name is required")
                if not request.region.coordinates or len(request.region.coordinates) == 0:
                    raise HTTPException(status_code=400, detail="Coordinates are required")
                lat, lon = request.region.coordinates[0]
                # print(f"Using coordinates for city {request.region.name}: lat={lat}, lon={lon}")
            elif request.region.region_type == 'custom':
                if not request.region.coordinates or len(request.region.coordinates) == 0:
                    raise HTTPException(status_code=400, detail="Coordinates are required for custom region")
                lat, lon = request.region.coordinates[0]
                # print(f"Using custom coordinates: lat={lat}, lon={lon}")
            else:
                raise HTTPException(status_code=400, detail="Invalid region type. Must be 'city' or 'custom'")

            # print(f"Analyzing region: lat={lat}, lon={lon}, start_year={request.start_year}-01-01, end_year={request.end_year}-01-01, hazard_type={request.hazard_type}")


            # Process the analysis request using WeatherProcessor
            weather_processor = WeatherProcessor()
            try:
                analysis_result = await weather_processor.analyze_region(
                    lat=lat,
                    lon=lon,
                    start_year=f"{request.start_year}-01-01",
                    end_year=f"{request.end_year}-01-01",
                    hazard_type=request.hazard_type
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Weather analysis failed: {str(e)}")

            # Extract time series data from yearly statistics
            yearly_stats = analysis_result.get('summary', {}).get('yearly_statistics', {})
            time_series_data = yearly_stats.get('intensities', [])
            if not time_series_data:
                time_series_data = yearly_stats.get('frequencies', [])

            if not time_series_data:
                raise HTTPException(status_code=500, detail="No time series data available for statistical analysis")

            # Perform advanced statistical analysis
            statistical_results = StatisticalAnalysis.comprehensive_analysis(time_series_data)

            result = {
                "region": request.region.dict(),
                "hazard_type": request.hazard_type,
                "period": f"{request.start_year}-{request.end_year}",
                "trends": {
                    "frequency": analysis_result['trends']['trend_coefficient'],
                    "intensity": analysis_result['trends']['percent_change'] / 100,
                    "duration": 0.0
                },
                "summary": analysis_result['summary'],
                "yearly_statistics": yearly_stats,
                "statistical_analysis": statistical_results
            }

            # Store results in MongoDB
            db_handler = Database()
            inserted_result = await db_handler.save_analysis(result)
            
            # Return the result with the converted ObjectId
            return AnalysisResponse(
                id=str(inserted_result.inserted_id),
                **result
            )
        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/summary/{region_id}")
@cache(expire=3600)  # Cache for 1 hour
async def get_summary(
    region_id: str,
    hazard_type: str = Query(..., enum=["heatwave", "drought", "rainfall"])
):
    try:
        if not ObjectId.is_valid(region_id):
            raise HTTPException(status_code=400, detail="Invalid region ID format")

        result = await db.analysis_results.find_one({"_id": ObjectId(region_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Convert ObjectId to string in the result
        result["id"] = str(result.pop("_id"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualization-data/{region_id}")
@cache(expire=3600)
async def get_visualization_data(
    region_id: str,
    hazard_type: str = Query(..., enum=["heatwave", "drought", "rainfall"])
):
    try:
        if not ObjectId.is_valid(region_id):
            raise HTTPException(status_code=400, detail="Invalid region ID format")

        analysis_data = await db.analysis_results.find_one({"_id": ObjectId(region_id)})
        if not analysis_data:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Use the yearly_statistics directly from the analysis result
        yearly_stats = analysis_data.get('yearly_statistics', {})
        if not yearly_stats:
            raise HTTPException(status_code=404, detail="No yearly statistics found")

        years = yearly_stats.get('years', [])
        frequencies = yearly_stats.get('frequencies', [])
        intensities = yearly_stats.get('intensities', [])

        # Create visualization datasets
        visualization_data = {
            "labels": [str(year) for year in years],
            "datasets": [
                {
                    "label": f"{hazard_type.capitalize()} Events Count",
                    "data": frequencies,
                    "borderColor": 'rgb(75, 192, 192)',
                    "tension": 0.1
                },
                {
                    "label": f"{hazard_type.capitalize()} Intensity",
                    "data": intensities,
                    "borderColor": 'rgb(255, 99, 132)',
                    "tension": 0.1
                }
            ]
        }
        
        return visualization_data
        
    except Exception as e:
        print(f"Error in get_visualization_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/{region_id}")
async def export_data(
    region_id: str,
    hazard_type: str = Query(..., enum=["heatwave", "drought", "rainfall"]),
    format: str = Query("excel", enum=["csv", "excel"])
):
    try:
        print(f"Attempting to export data for region_id: {region_id}, hazard_type: {hazard_type}")
        
        if not ObjectId.is_valid(region_id):
            raise HTTPException(status_code=400, detail="Invalid region ID format")

        # Query MongoDB directly first
        result = await db.analysis_results.find_one({"_id": ObjectId(region_id)})
        print(f"Direct MongoDB query result: {result is not None}")

        if not result:
            # Try querying by string ID as fallback
            result = await db.analysis_results.find_one({"id": region_id})
            print(f"Fallback query result: {result is not None}")

        if not result:
            raise HTTPException(status_code=404, detail=f"Analysis not found for ID: {region_id}")

        try:
            # Create export handler and format data
            export_handler = ExportHandler()
            formatted_data = export_handler.format_data_for_export(result)
            
            # Generate Excel file
            buffer = export_handler.create_excel(formatted_data)
            
            # Set appropriate filename extension and media type
            extension = "xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            return StreamingResponse(
                buffer,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="climate_analysis_{region_id}.{extension}"'
                }
            )
        except Exception as e:
            print(f"Error during export generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating export: {str(e)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error during export: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends/{region_id}")
async def get_trends(
    region_id: str,
    hazard_type: str = Query(..., enum=["heatwave", "drought", "rainfall"])
):
    try:
        if not ObjectId.is_valid(region_id):
            raise HTTPException(status_code=400, detail="Invalid region ID format")

        # Retrieve analysis data
        db_handler = Database()
        analysis_data = await db_handler.get_analysis_history(region_id)

        if not analysis_data:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Extract time series data
        time_series = [point['value'] for point in analysis_data['data_points']]
        
        # Perform Mann-Kendall test and comprehensive analysis
        trend_analysis = StatisticalAnalysis.comprehensive_analysis(time_series)

        return {
            "trend_analysis": trend_analysis,
            "region_id": region_id,
            "hazard_type": hazard_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-weather")
async def test_weather_api():
    try:
        # Test with London coordinates
        lat, lon = 51.5074, -0.1278
        current_year = datetime.now().year
        # Use last year and current year up to today
        start_year = current_year - 4
        end_year = current_year -1
        # print(f"Testing with coordinates: lat={lat}, lon={lon}")
        # print(f"Date range: {start_year} to {end_year}")
        
        weather_processor = WeatherProcessor()
        df = await weather_processor.fetch_weather_data(lat, lon, start_year, end_year)
        return {
            "status": "success",
            "message": "Weather API connection successful",
            "data_sample": df.head().to_dict(),
            "date_range": {
                "start": start_year,
                "end": end_year
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
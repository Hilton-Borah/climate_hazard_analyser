from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from datetime import datetime
import os
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://Hilton:hilton@cluster0.rgtirz5.mongodb.net/climate_hazards?retryWrites=true&w=majority")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.climate_hazards

class Database:
    @staticmethod
    async def save_analysis(analysis_data: dict):
        """Save analysis results to MongoDB"""
        try:
            analysis_data["timestamp"] = datetime.utcnow()
            result = await db.analysis_results.insert_one(analysis_data)
            print(f"Saved analysis with ID: {result.inserted_id}")
            return result
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")
            raise

    @staticmethod
    async def get_analysis_history(region_id: str = None):
        """Retrieve analysis by ID or get history"""
        try:
            if region_id:
                # Try ObjectId first
                if ObjectId.is_valid(region_id):
                    result = await db.analysis_results.find_one({"_id": ObjectId(region_id)})
                    if result:
                        print(f"Found analysis by ObjectId: {region_id}")
                        return result
                
                # Try string ID as fallback
                result = await db.analysis_results.find_one({"id": region_id})
                if result:
                    print(f"Found analysis by string ID: {region_id}")
                    return result
                
                print(f"No analysis found for ID: {region_id}")
                return None
            else:
                # Get latest analyses
                cursor = db.analysis_results.find().sort("timestamp", DESCENDING).limit(10)
                results = await cursor.to_list(length=10)
                print(f"Retrieved {len(results)} recent analyses")
                return results
        except Exception as e:
            print(f"Error retrieving analysis: {str(e)}")
            raise

    @staticmethod
    async def save_hazard_data(hazard_data: dict):
        """Save hazard detection data"""
        try:
            hazard_data["timestamp"] = datetime.utcnow()
            result = await db.hazard_data.insert_one(hazard_data)
            print(f"Saved hazard data with ID: {result.inserted_id}")
            return result
        except Exception as e:
            print(f"Error saving hazard data: {str(e)}")
            raise

    @staticmethod
    async def get_hazard_statistics(region: str, hazard_type: str = None):
        """Get hazard statistics for a region"""
        try:
            match_stage = {"$match": {"region": region}}
            if hazard_type:
                match_stage["$match"]["hazard_type"] = hazard_type

            pipeline = [
                match_stage,
                {
                    "$group": {
                        "_id": "$hazard_type",
                        "count": {"$sum": 1},
                        "avg_intensity": {"$avg": "$intensity"},
                        "last_occurrence": {"$max": "$timestamp"}
                    }
                }
            ]
            
            results = await db.hazard_data.aggregate(pipeline).to_list(None)
            print(f"Retrieved hazard statistics for region {region}: {len(results)} results")
            return results
        except Exception as e:
            print(f"Error retrieving hazard statistics: {str(e)}")
            raise 
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import nest_asyncio
import uvicorn

# Enable FastAPI to run in Colab
nest_asyncio.apply()

# Load your hotel booking dataset
df = pd.read_csv("hotel_bookings.csv")  # Load your dataset

app = FastAPI()

class AnalyticsRequest(BaseModel):
    report_type: str  # e.g., "revenue_trends", "cancellation_rate"

# Define analytics functions
def revenue_trends():
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    return df.groupby(df['arrival_date'].dt.to_period('M'))['revenue'].sum().to_dict()

def cancellation_rate():
    total_bookings = len(df)
    cancelled_bookings = df['is_canceled'].sum()
    return {"cancellation_rate": (cancelled_bookings / total_bookings) * 100}

def geo_distribution():
    return df['country'].value_counts().to_dict()

def booking_lead_time():
    return df['lead_time'].describe().to_dict()

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    if request.report_type == "revenue_trends":
        return revenue_trends()
    elif request.report_type == "cancellation_rate":
        return cancellation_rate()
    elif request.report_type == "geo_distribution":
        return geo_distribution()
    elif request.report_type == "booking_lead_time":
        return booking_lead_time()
    elif request.report_type == "all":
        return {
            "revenue_trends": revenue_trends(),
            "cancellation_rate": cancellation_rate(),
            "geo_distribution": geo_distribution(),
            "booking_lead_time": booking_lead_time()
        }    
    else:
        return {"error": "Invalid report type"}

# Run the API
uvicorn.run(app, host="0.0.0.0", port=8000)

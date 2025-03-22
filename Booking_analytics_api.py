from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import nest_asyncio
import uvicorn

# Enable FastAPI to run in Colab (if needed)
nest_asyncio.apply()

# Load your hotel booking dataset
df = pd.read_csv("hotel_bookings.csv")  # Ensure this file exists in your working directory

app = FastAPI()

class AnalyticsRequest(BaseModel):
    report_type: str  # e.g., "revenue_trends", "cancellation_rate", "geo_distribution", "booking_lead_time", "all"

# Define analytics functions
def revenue_trends():
    # Compute a dummy revenue if the 'revenue' column doesn't exist.
    # Here we assume a fixed price per night.
    price_per_night = 100  # adjust as needed
    df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    df["revenue"] = df["total_nights"] * price_per_night

    # Create 'arrival_date' column if not present.
    if "arrival_date" not in df.columns:
        try:
            df["arrival_date"] = pd.to_datetime(
                df["arrival_date_year"].astype(str) + " " +
                df["arrival_date_month"] + " " +
                df["arrival_date_day_of_month"].astype(str),
                format="%Y %B %d",
                errors="coerce"
            )
        except Exception as e:
            raise KeyError("Error constructing 'arrival_date' column: " + str(e))
    
    # Group by month (as a period) and sum revenue.
    result = df.groupby(df["arrival_date"].dt.to_period("M"))["revenue"].sum().to_dict()
    # Convert Period keys to string
    return {str(k): v for k, v in result.items()}

def cancellation_rate():
    total_bookings = len(df)
    cancelled_bookings = df["is_canceled"].sum()
    return {"cancellation_rate": (cancelled_bookings / total_bookings) * 100}

def geo_distribution():
    return df["country"].value_counts().to_dict()

def booking_lead_time():
    return df["lead_time"].describe().to_dict()

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

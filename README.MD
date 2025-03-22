# LLM-Powered-Booking-Analytics-QA-System

This repository contains multiple projects designed to work with hotel booking data. The projects include a Streamlit application, FastAPI endpoints for querying data and analytics, and a Jupyter Notebook for data preprocessing and exploratory analytics.

## Repository Structure

- **Data_preprocessing_and_analysis.ipynb**  
  A Jupyter Notebook that includes data preprocessing tasks such as:
  - Filling missing values
  - Removing duplicates and handling inconsistencies
  - Performing exploratory data analytics on the hotel booking dataset

- **App.py**  
  A [Streamlit](https://streamlit.io/) LLM powered application(using RAG) that allows you to interact with hotel booking data uploaded as a CSV file. Use this interactive chatbot to solve your query.

- **Booking_query_api.py**  
  A FastAPI-based service that provides an API endpoint for querying hotel booking data. Post your queries to this API to receive responses based on the underlying data analysis.

- **Booking_analytics_api.py**  
  Another FastAPI service that exposes endpoints to retrieve specific analytics or all available analytics computed on the dataset. This service lets you fetch insights programmatically.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.10+
- [Streamlit](https://docs.streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/) and a compatible ASGI server (e.g., [uvicorn](https://www.uvicorn.org/))
- Jupyter Notebook (if you plan to run the assignment notebook)

You can install the Python dependencies using pip:

```bash
pip install -r requirements.txt 
```

## 🤖Setup

### Clone the repository

```bash
git clone https://github.com/siddhant1721/-LLM-Powered-Booking-Analytics-QA-System-
```
## ▶️▶️Running the Applications

### 📅Data_preprocessing_and_analysis
run the jupyter notebook

### 💻Streamlit LLM App

To launch the Streamlit app:

```bash
streamlit run "path to your file"
```
### 📃Booking Analytics API

This API processes hotel booking data and provides various analytics reports.

#### Getting Started

1. **Start the Server:**  
   Run the Python file to start the FastAPI server. For example:
   ```sh
   python Booking_analytics_api.py
   ```
2. You should see output indicating the server is running on `http://0.0.0.0:8000`.

   #### Test the API Endpoints Using cURL

      Make sure you have cURL installed. Open your command prompt and use the following commands:

   ### For All Reports:
      ```sh
      curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"report_type\": \"all\"}"
      ```   
   ### For Revenue Trends:
      ```sh
      curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"report_type\": \"revenue_trends\"}"
      ```  
   ### For Cancellation Rates:
      ```sh
     curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"report_type\": \"cancellation_rate\"}"
      ```  

   ### For Geographical Distribution:
      ```sh
     curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"report_type\": \"geo_distribution\"}"
      ```  
   ### For Booking Lead Time:
      ```sh
    curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"report_type\": \"booking_lead_time\"}"
      ```              


### 🤔Booking_query_api

1. Start the Server

      Run this command in your terminal (make sure to change to the correct directory):

      ```sh
      uvicorn Booking_query_api:app --host 0.0.0.0 --port 8000
      ```
2. Query the api

    ```sh
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the cancellation rate?\"}"
    ```





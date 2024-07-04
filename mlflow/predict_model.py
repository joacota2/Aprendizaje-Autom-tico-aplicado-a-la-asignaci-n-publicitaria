import requests
from pydantic import BaseModel

class MediaData(BaseModel):
    Paid_Views: float
    Google_Impressions: float
    Email_Impressions: float
    Facebook_Impressions: float
    Affiliate_Impressions: float
    is_peak_season: int

def make_prediction(media_data: MediaData):
    url = "http://127.0.0.1:1234/invocations"
    data = {
        "columns": ["Paid_Views", "Google_Impressions", "Email_Impressions", "Facebook_Impressions", "Affiliate_Impressions", "is_peak_season"],
        "data": [[media_data.Paid_Views, media_data.Google_Impressions, media_data.Email_Impressions, media_data.Facebook_Impressions, media_data.Affiliate_Impressions, media_data.is_peak_season]]
    }
    response = requests.post(url, json=data)
    return response.json()

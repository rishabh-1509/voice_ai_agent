import aiohttp
from config import settings

class WeatherService:
    async def get_weather(self, location: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{settings.weather_api_base}?q={location}&appid={settings.weather_api_key}&units=metric"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return {"error": "Weather data not available"}
        except Exception as e:
            return {"error": str(e)}
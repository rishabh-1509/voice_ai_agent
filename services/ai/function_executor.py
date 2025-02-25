from services.integrations.calendar import CalendarIntegration
from services.integrations.weather import WeatherService

class FunctionExecutor:
    def __init__(self):
        self.calendar = CalendarIntegration()
        self.weather = WeatherService()

    async def execute(self, intent: str, params: dict) -> str:
        handler = getattr(self, f"handle_{intent}", self.handle_default)
        return await handler(params)

    async def handle_book_appointment(self, params: dict) -> str:
        # Implementation using calendar service
        pass

    async def handle_check_weather(self, params: dict) -> str:
        weather_data = await self.weather.get_weather(params.get("location", ""))
        if "error" in weather_data:
            return "Sorry, I couldn't retrieve the weather information."
        return f"Current temperature: {weather_data['main']['temp']}°C"

    async def handle_default(self, params: dict) -> str:
        return "I'm sorry, I didn't understand that request."
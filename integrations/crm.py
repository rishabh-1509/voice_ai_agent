from tenacity import retry, stop_after_attempt
from config import settings
from utilities.logger import logger

class CRMClient:
    def __init__(self):
        self.base_url = settings.CRM_API_URL
        self.api_key = settings.CRM_API_KEY

    @retry(stop=stop_after_attempt(3))
    async def get_user_profile(self, user_id: str) -> dict:
        """Fetch user profile from CRM system"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(
                    f"{self.base_url}/users/{user_id}",
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"CRM Error: {str(e)}")
            return {"error": "CRM system unavailable"}
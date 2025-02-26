from google.oauth2 import service_account
from googleapiclient.discovery import build
from tenacity import retry, stop_after_attempt
from config import settings
from utilities.logger import logger

class CalendarManager:
    def __init__(self):
        self.service = self._authenticate()
    
    def _authenticate(self):
        credentials = service_account.Credentials.from_service_account_file(
            'credentials.json',
            scopes=['https://www.googleapis.com/auth/calendar']
        )
        return build('calendar', 'v3', credentials=credentials)

    @retry(stop=stop_after_attempt(3))
    async def create_event(self, event_details: dict) -> str:
        event = self.service.events().insert(
            calendarId='primary',
            body=event_details
        ).execute()
        logger.info(f"Created event: {event['id']}")
        return event['id']
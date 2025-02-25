from googleapiclient.discovery import build
from config import settings
import datetime
import asyncio

class CalendarIntegration:
    def __init__(self):
        self.service = build('calendar', 'v3', developerKey=settings.calendar_api_key)
        
    async def create_event(self, event_details):
        try:
            event = {
                'summary': event_details['summary'],
                'start': {'dateTime': event_details['start_time']},
                'end': {'dateTime': event_details['end_time']},
            }
            return await asyncio.to_thread(
                lambda: self.service.events().insert(
                    calendarId='primary',
                    body=event
                ).execute()
            )
        except Exception as e:
            print(f"Calendar error: {e}")
            return None
            
    async def delete_event(self, event_id):
        try:
            return await asyncio.to_thread(
                lambda: self.service.events().delete(
                    calendarId='primary',
                    eventId=event_id
                ).execute()
            )
        except Exception as e:
            print(f"Calendar deletion error: {e}")
            return False
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

class CalendarIntegration:
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self, token_path='token.json', creds_path='credentials.json'):
        self.creds = self._authenticate(token_path, creds_path)
        self.service = build('calendar', 'v3', credentials=self.creds)

    def _authenticate(self, token_path, creds_path):
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
                
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                
        return creds
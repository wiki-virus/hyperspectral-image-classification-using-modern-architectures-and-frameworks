import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/documents']

def main():
    """Shows basic usage of the Docs API.
    Creates a new Google Doc and prints its document ID.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('docs', 'v1', credentials=creds)

        # The title of the document to create
        document = {'title': 'My New Google Doc via API'}

        # Create the document
        doc = service.documents().create(body=document).execute()
        
        print(f"Created document with title: {doc.get('title')}")
        print(f"Document ID: {doc.get('documentId')}")
        print(f"You can view it here: https://docs.google.com/document/d/{doc.get('documentId')}/edit")

    except HttpError as err:
        print(err)

if __name__ == '__main__':
    main()

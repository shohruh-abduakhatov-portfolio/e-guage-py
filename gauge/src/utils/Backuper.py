from __future__ import print_function

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# todo finish here, so that daily captured gauge imgs are backed up
"""https://developers.google.com/drive/api/v3/resumable-upload#example_resuming_an_interrupted_upload"""

"""
    Client ID
    408031432420-1698nqmcb9aau0gfq8kpuhu5gv62ekso.apps.googleusercontent.com
    
    Client Secret
    7rz9y60obkHF7IQr6dVm-HBz
"""


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print('{0} ({1})'.format(item['name'], item['id']))


if __name__ == '__main__':
    main()

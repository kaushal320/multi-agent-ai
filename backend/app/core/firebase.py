import json
import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

from app.core.config import settings

# Support both file path (Docker secret) and JSON string (env var for Render/other platforms)
if settings.firebase_service_account_json:
    cred_dict = json.loads(settings.firebase_service_account_json)
    cred = credentials.Certificate(cred_dict)
else:
    cred = credentials.Certificate(settings.firebase_credentials_path)

firebase_app = firebase_admin.initialize_app(cred)


def verify_id_token(token: str) -> dict:
    """Verifies a Firebase ID token sent from the frontend and returns its decoded claims.
    A small clock-skew allowance absorbs minor drift between this machine's clock and
    Firebase's servers."""
    return firebase_auth.verify_id_token(token, clock_skew_seconds=30)

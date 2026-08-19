from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_uri: str
    mongo_db_name: str

    redis_url: str

    firebase_credentials_path: str = "/run/secrets/firebase_service_account"
    firebase_service_account_json: str = ""

    frontend_url: str

    session_cookie_name: str
    session_ttl_seconds: int

    environment: str
    logfire_token: str
    logfire_base_url: str

    groq_api_key: str
    google_api_key: str
    tavily_api_key: str
    qdrant_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

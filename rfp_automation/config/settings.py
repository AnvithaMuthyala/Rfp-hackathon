from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

env_path = Path.cwd() / ".env"


class Settings(BaseSettings):

    TAVILY_API_KEY: str
    GOOGLE_API_KEY: str
    CHROMA_PERSIST_DIRECTORY: str
    LOG_LEVEL: str = "INFO"
    STREAMLIT_SERVER_PORT: int

    AZURE_DEPLOYMENT: str
    AZURE_API_VERSION: str
    AZURE_ENDPOINT: str
    AZURE_API_KEY: str
    AZURE_MODEL: str

    model_config = SettingsConfigDict(
        env_file=str(env_path), extra="allow", env_file_encoding="utf-8"
    )


get_settings = lambda: Settings()

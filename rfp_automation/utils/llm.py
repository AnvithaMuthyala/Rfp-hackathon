from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from ..config.settings import get_settings

settings = get_settings()


def get_llm(
    temperature: float = 0.7,
    max_tokens: int = 1024,
    response_model: type[BaseModel] | None = None,
):

    client = AzureChatOpenAI(
        azure_deployment=settings.AZURE_DEPLOYMENT,
        api_version=settings.AZURE_API_VERSION,
        azure_endpoint=settings.AZURE_ENDPOINT,
        api_key=settings.AZURE_API_KEY,  # type:ignore
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
    )
    if response_model:
        client = client.with_structured_output(response_model)

    def generate(prompt: str):
        response = client.invoke(prompt)
        return response

    return generate

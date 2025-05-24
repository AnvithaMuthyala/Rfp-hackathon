from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

import chromadb
import chromadb.utils.embedding_functions as ef
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config.settings import get_settings
from ..workflow.state import EnhancedRFPState

settings = get_settings()


class KnowledgeManagementAgent:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = ef.GoogleGenerativeAiEmbeddingFunction(
            api_key=settings.GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.collection_name = "rfp_knowledge_base"
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize ChromaDB with persistence"""

        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            return client.get_collection(self.collection_name)
        except ValueError:
            return client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,  # type:ignore
            )

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Retrieve relevant knowledge and cache new information"""
        parsed = state["parsed_requirements"]
        market_research = state.get("market_research", {})

        # Search existing knowledge base
        query = f"{parsed.get('domain')} {parsed.get('scale')} users requirements"
        cached_results = self._search_knowledge_base(query)

        # Cache new market research data
        if market_research and not market_research.get("error"):
            self._cache_market_research(market_research, parsed)

        state["cached_knowledge"] = cached_results
        return state

    def _search_knowledge_base(self, query: str, k: int = 3) -> List[Dict]:
        """Search ChromaDB for relevant cached knowledge"""
        try:
            if self.vector_store.count() == 0:
                return []

            results = self.vector_store.query(query_texts=query, n_results=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        "content": doc,
                        "relevance_score": float(score),
                        "source": "knowledge_base",
                    }
                )

            return formatted_results
        except Exception as e:
            return []

    def _cache_market_research(self, market_research: Dict, requirements: Dict):
        """Cache market research data in ChromaDB"""
        try:
            documents = []

            # Cache market trends
            for trend in market_research.get("market_trends", []):
                doc = Document(
                    page_content=trend,
                    metadata={
                        "type": "market_trend",
                        "domain": requirements.get("domain"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "tavily_search",
                    },
                )
                documents.append(doc)

            # Cache vendor insights
            for insight in market_research.get("vendor_landscape", []):
                doc = Document(
                    page_content=insight,
                    metadata={
                        "type": "vendor_insight",
                        "domain": requirements.get("domain"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "tavily_search",
                    },
                )
                documents.append(doc)

            if documents:
                # Split long documents
                split_docs = self.text_splitter.split_documents(documents)
                self.vector_store.add(
                    ids=[str(uuid4()) for x in range(len(split_docs))],
                    documents=list(map(lambda doc: doc.page_content, split_docs)),
                )

        except Exception as e:
            # Log error but don't fail the workflow
            pass

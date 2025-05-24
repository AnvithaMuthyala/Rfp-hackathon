from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as ef
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import TavilySearchAPIRetriever

from ..config.settings import get_settings
from ..workflow.state import EnhancedRFPState

settings = get_settings()


data_folder = Path.cwd() / "data"
data_folder.mkdir(exist_ok=True)


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
        self.relevance_threshold = 0.8
        self.tavily_retriever = TavilySearchAPIRetriever()

        self.index_documents_from_folder(str(data_folder))

    def _initialize_vector_store(self):
        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            return client.get_collection(self.collection_name)
        except ValueError:
            return client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,  # type:ignore
            )

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        parsed = state["parsed_requirements"]
        domains = parsed.get("domain", [])
        scale = parsed.get("scale", "")
        market_research = state.get("market_research", {})

        cached_knowledge = []
        print("domains", domains)

        for domain in domains:
            query = f"{domain} {scale} users requirements"
            results = self._search_knowledge_base(query)

            print("results", results)
            if not results or results[0]["relevance_score"] < self.relevance_threshold:
                tavily_results = self._fetch_tavily_data(query)
                if tavily_results:
                    self._cache_market_research(tavily_results, {"domain": domain})
                    results = self._search_knowledge_base(query)

            cached_knowledge.extend(results)

        state["cached_knowledge"] = cached_knowledge
        return state

    def _search_knowledge_base(self, query: str, k: int = 3) -> List[Dict]:
        try:
            if self.vector_store.count() == 0:
                return []

            results = self.vector_store.query(query_texts=[query], n_results=k)

            formatted_results = []
            for doc, score in zip(
                results.get("documents")[0], results["distances"][0]  # type:ignore
            ):
                formatted_results.append(
                    {
                        "content": doc,
                        "relevance_score": float(score),
                        "source": "knowledge_base",
                    }
                )

            return formatted_results
        except Exception:
            return []

    def _fetch_tavily_data(self, query: str) -> Dict:
        try:
            search_results = self.tavily_retriever.get_relevant_documents(query)
            market_trends = [doc.page_content for doc in search_results]
            return {"market_trends": market_trends}
        except Exception:
            return {}

    def _cache_market_research(self, market_research: Dict, requirements: Dict):
        try:
            documents = []

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
                split_docs = self.text_splitter.split_documents(documents)
                self.vector_store.add(
                    ids=[str(uuid4()) for _ in range(len(split_docs))],
                    documents=[doc.page_content for doc in split_docs],
                )
        except Exception:
            pass

    def index_documents_from_folder(self, base_path: str):
        base_dir = Path(base_path)
        if not base_dir.exists() or not base_dir.is_dir():
            print(f"Directory {base_path} does not exist or is not a directory.")
            return

        for domain_dir in base_dir.iterdir():
            if domain_dir.is_dir():
                domain = domain_dir.name
                for txt_file in domain_dir.glob("*.txt"):
                    try:
                        content = txt_file.read_text(encoding="utf-8").strip()
                        doc = Document(
                            page_content=content,
                            metadata={
                                "type": "domain_knowledge",
                                "domain": domain,
                                "timestamp": datetime.now().isoformat(),
                                "source": str(txt_file),
                            },
                        )
                        split_docs = self.text_splitter.split_documents([doc])
                        self.vector_store.add(
                            ids=[str(uuid4()) for _ in range(len(split_docs))],
                            documents=[doc.page_content for doc in split_docs],
                        )
                        print(f"Indexed {txt_file} under domain '{domain}'.")
                    except Exception as e:
                        print(f"Failed to read {txt_file}: {e}")

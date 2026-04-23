"""
SynapseKit — lightweight, async-first RAG framework.

3-line happy path:

    from synapsekit import RAG

    rag = RAG(model="gpt-4o-mini", api_key="sk-...")
    rag.add("Your document text here")

    async for token in rag.stream("What is the main topic?"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from ._api import deprecated, experimental, public_api
from .a2a import A2AClient, A2AMessage, A2AServer, A2ATask, AgentCard, TaskState
from .agents import (
    ActionEvent,
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentScratchpad,
    AgentStep,
    ArxivSearchTool,
    BaseTool,
    BingSearchTool,
    BraveSearchTool,
    BrowserTool,
    CalculatorTool,
    CodeInterpreterTool,
    ContentFilter,
    Crew,
    CrewAgent,
    CrewResult,
    DateTimeTool,
    DuckDuckGoSearchTool,
    EmailTool,
    ErrorEvent,
    FileListTool,
    FileReadTool,
    FileWriteTool,
    FinalAnswerEvent,
    FunctionCallingAgent,
    GitHubAPITool,
    GoogleSearchTool,
    GraphQLTool,
    GuardrailResult,
    Guardrails,
    Handoff,
    HandoffChain,
    HandoffResult,
    HTTPRequestTool,
    HumanInputTool,
    ImageAnalysisTool,
    JiraTool,
    JSONQueryTool,
    LinearTool,
    NewsTool,
    ObservationEvent,
    PDFReaderTool,
    PIIDetector,
    PIIRedactor,
    PubMedSearchTool,
    PythonREPLTool,
    ReActAgent,
    RedactionResult,
    RegexTool,
    SentimentAnalysisTool,
    ShellTool,
    SimpleAgent,
    SlackTool,
    SpeechToTextTool,
    SQLQueryTool,
    SQLSchemaInspectionTool,
    StepEvent,
    StripeTool,
    SummarizationTool,
    SupervisorAgent,
    Task,
    TavilySearchTool,
    TextToSpeechTool,
    ThoughtEvent,
    TokenEvent,
    ToolRegistry,
    ToolResult,
    TopicRestrictor,
    TranslationTool,
    TwilioTool,
    VectorSearchTool,
    WeatherTool,
    WebScraperTool,
    WebSearchTool,
    WikipediaTool,
    WolframAlphaTool,
    WorkerAgent,
    YouTubeSearchTool,
    agent,
    tool,
)
from .embeddings.backend import SynapsekitEmbeddings
from .evaluation import (
    EvalCaseMeta,
    EvalRegression,
    EvalSnapshot,
    EvaluationPipeline,
    EvaluationResult,
    FaithfulnessMetric,
    GroundednessMetric,
    MetricDelta,
    MetricResult,
    RegressionReport,
    RelevancyMetric,
    eval_case,
)
from .graph import (
    END,
    BaseCheckpointer,
    CompiledGraph,
    ConditionalEdge,
    ConditionFn,
    Edge,
    EventHooks,
    ExecutionTrace,
    GraphConfigError,
    GraphEvent,
    GraphInterrupt,
    GraphRuntimeError,
    GraphState,
    GraphVisualizer,
    InMemoryCheckpointer,
    InterruptState,
    JSONFileCheckpointer,
    Node,
    NodeFn,
    RecursionDepthError,
    SQLiteCheckpointer,
    StateField,
    StateGraph,
    TraceEntry,
    TypedState,
    agent_node,
    approval_node,
    dynamic_route_node,
    fan_out_node,
    get_mermaid_with_trace,
    llm_node,
    rag_node,
    sse_stream,
    subgraph_node,
    ws_stream,
)
from .llm.base import BaseLLM, LLMConfig
from .llm.cost_router import QUALITY_TABLE, CostRouter, CostRouterConfig, RouterModelSpec
from .llm.fallback_chain import FallbackChain, FallbackChainConfig
from .llm.multimodal import AudioContent, ImageContent, MultimodalMessage
from .llm.structured import generate_structured
from .loaders.arxiv import ArXivLoader
from .loaders.azure_blob import AzureBlobLoader
from .loaders.base import Document
from .loaders.bigquery import BigQueryLoader
from .loaders.config import ConfigLoader
from .loaders.confluence import ConfluenceLoader
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.dynamodb import DynamoDBLoader
from .loaders.email import EmailLoader
from .loaders.epub import EPUBLoader
from .loaders.gcs import GCSLoader
from .loaders.git import GitLoader
from .loaders.github import GitHubLoader
from .loaders.google_sheets import GoogleSheetsLoader
from .loaders.html import HTMLLoader
from .loaders.hubspot import HubSpotLoader
from .loaders.image import ImageLoader
from .loaders.jira import JiraLoader
from .loaders.json_loader import JSONLoader
from .loaders.latex import LaTeXLoader
from .loaders.markdown import MarkdownLoader
from .loaders.mongodb import MongoDBLoader
from .loaders.obsidian import ObsidianLoader
from .loaders.onedrive import OneDriveLoader
from .loaders.pdf import PDFLoader
from .loaders.rss import RSSLoader
from .loaders.rtf import RTFLoader
from .loaders.s3 import S3Loader
from .loaders.salesforce import SalesforceLoader
from .loaders.sql import SQLLoader
from .loaders.teams import TeamsLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.tsv import TSVLoader
from .loaders.web import WebLoader
from .loaders.wikipedia import WikipediaLoader
from .mcp import MCPClient, MCPServer, MCPToolAdapter
from .memory import AgentMemory as PersistentAgentMemory
from .memory.buffer import BufferMemory
from .memory.conversation import ConversationMemory
from .memory.entity import EntityMemory
from .memory.hybrid import HybridMemory
from .memory.redis import RedisConversationMemory
from .memory.sqlite import SQLiteConversationMemory
from .memory.summary_buffer import SummaryBufferMemory
from .memory.token_buffer import TokenBufferMemory
from .observability import (
    AuditEntry,
    AuditLog,
    BudgetExceededError,
    BudgetGuard,
    BudgetLimit,
    CircuitState,
    CostRecord,
    CostTracker,
    DistributedTracer,
    OTelExporter,
    Span,
    TraceSpan,
    TracingMiddleware,
    TracingUI,
)
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .plugins import PluginRegistry
from .prompts.hub import PromptHub
from .prompts.template import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .rag.facade import RAG
from .rag.pipeline import RAGConfig, RAGPipeline
from .retrieval.adaptive import AdaptiveRAGRetriever
from .retrieval.base import VectorStore
from .retrieval.cohere_reranker import CohereReranker
from .retrieval.contextual import ContextualRetriever
from .retrieval.contextual_compression import ContextualCompressionRetriever
from .retrieval.crag import CRAGRetriever
from .retrieval.cross_encoder import CrossEncoderReranker
from .retrieval.ensemble import EnsembleRetriever
from .retrieval.flare import FLARERetriever
from .retrieval.graphrag import GraphRAGRetriever, KnowledgeGraph
from .retrieval.hybrid_search import HybridSearchRetriever
from .retrieval.hyde import HyDERetriever
from .retrieval.mongodb_atlas import MongoDBAtlasVectorStore
from .retrieval.multi_step import MultiStepRetriever
from .retrieval.parent_document import ParentDocumentRetriever
from .retrieval.query_decomposition import QueryDecompositionRetriever
from .retrieval.rag_fusion import RAGFusionRetriever
from .retrieval.retriever import Retriever
from .retrieval.self_query import SelfQueryRetriever
from .retrieval.self_rag import SelfRAGRetriever
from .retrieval.sentence_window import SentenceWindowRetriever
from .retrieval.step_back import StepBackRetriever
from .retrieval.vectorstore import InMemoryVectorStore
from .text_splitters import (
    BaseSplitter,
    CharacterTextSplitter,
    CodeSplitter,
    HTMLTextSplitter,
    JSONSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    SentenceTextSplitter,
    SentenceWindowSplitter,
    TokenAwareSplitter,
)

__version__ = "1.5.6"
__all__ = [
    # Graph workflows
    "END",
    "QUALITY_TABLE",
    # Facade
    "RAG",
    "VLLMLLM",
    # A2A Protocol
    "A2AClient",
    "A2AMessage",
    "A2AServer",
    "A2ATask",
    # Step events
    "ActionEvent",
    "AdaptiveRAGRetriever",
    "AgentCard",
    "AgentConfig",
    "AgentExecutor",
    "AgentMemory",
    "AgentScratchpad",
    "AgentStep",
    "AirtableLoader",
    "AlephAlphaLLM",
    "ArXivLoader",
    # Built-in tools
    "ArxivSearchTool",
    # LLM
    "AsyncLRUCache",
    # Multimodal
    "AudioContent",
    "AudioLoader",
    # Observability
    "AuditEntry",
    "AuditLog",
    "AzureBlobLoader",
    "AzureOpenAILLM",
    # Checkpointers
    "BaseCheckpointer",
    "BaseLLM",
    # Text splitters
    "BaseSplitter",
    # Agents
    "BaseTool",
    "BigQueryLoader",
    "BingSearchTool",
    "BraveSearchTool",
    "BrowserTool",
    "BudgetExceededError",
    "BudgetGuard",
    "BudgetLimit",
    "BufferMemory",
    "CRAGRetriever",
    "CSVLoader",
    "CalculatorTool",
    "CerebrasLLM",
    "CharacterTextSplitter",
    "ChatPromptTemplate",
    "ChromaVectorStore",
    "CircuitState",
    "CloudflareLLM",
    "CodeInterpreterTool",
    "CodeSplitter",
    "CohereReranker",
    "CompiledGraph",
    "ConditionFn",
    "ConditionalEdge",
    "ConfigLoader",
    "ConfluenceLoader",
    # Guardrails
    "ContentFilter",
    "ContextualCompressionRetriever",
    "ContextualRetriever",
    "ConversationMemory",
    "CostRecord",
    "CostRouter",
    "CostRouterConfig",
    # Cost intelligence
    "CostTracker",
    # Multi-agent
    "Crew",
    "CrewAgent",
    "CrewResult",
    "CrossEncoderReranker",
    "DateTimeTool",
    "DeepSeekLLM",
    "DirectoryLoader",
    "DiscordLoader",
    "DistributedTracer",
    # Loaders
    "Document",
    "DocxLoader",
    "DropboxLoader",
    "DuckDuckGoSearchTool",
    "DynamoDBCacheBackend",
    "DynamoDBLoader",
    "EPUBLoader",
    "Edge",
    "ElasticsearchLoader",
    "EmailLoader",
    "EmailTool",
    "EnsembleRetriever",
    "EntityMemory",
    "ErrorEvent",
    # Evaluation
    "EvalCaseMeta",
    "EvalRegression",
    "EvalSnapshot",
    "EvaluationPipeline",
    "EvaluationResult",
    "EventHooks",
    "ExcelLoader",
    "ExecutionTrace",
    "FAISSVectorStore",
    "FLARERetriever",
    "FaithfulnessMetric",
    "FallbackChain",
    "FallbackChainConfig",
    "FewShotPromptTemplate",
    "FileListTool",
    "FileReadTool",
    "FileWriteTool",
    "FinalAnswerEvent",
    "FireworksLLM",
    "FunctionCallingAgent",
    "GCSLoader",
    "GPT4AllLLM",
    "GitHubAPITool",
    "GitHubLoader",
    "GitLoader",
    "GoogleDriveLoader",
    "GoogleSearchTool",
    "GoogleSheetsLoader",
    "GraphConfigError",
    "GraphEvent",
    "GraphInterrupt",
    "GraphQLTool",
    "GraphRAGRetriever",
    "GraphRuntimeError",
    "GraphState",
    "GraphVisualizer",
    "GroqLLM",
    "GroundednessMetric",
    "GuardrailResult",
    "Guardrails",
    "HTMLLoader",
    "HTMLTextSplitter",
    "HTTPRequestTool",
    "Handoff",
    "HandoffChain",
    "HandoffResult",
    "HubSpotLoader",
    "HuggingFaceLLM",
    "HumanInputTool",
    "HyDERetriever",
    "HybridMemory",
    "HybridSearchRetriever",
    "ImageAnalysisTool",
    "ImageContent",
    "ImageLoader",
    "InMemoryCheckpointer",
    "InMemoryVectorStore",
    "InterruptState",
    "JSONFileCheckpointer",
    "JSONLoader",
    # Parsers
    "JSONParser",
    "JSONQueryTool",
    "JSONSplitter",
    "JiraLoader",
    "JiraTool",
    "KnowledgeGraph",
    "LLMConfig",
    "LMStudioLLM",
    "LaTeXLoader",
    "LanceDBVectorStore",
    "LinearTool",
    "ListParser",
    # MCP
    "MCPClient",
    "MCPServer",
    "MCPToolAdapter",
    "MarkdownLoader",
    "MarkdownTextSplitter",
    "MemcachedCacheBackend",
    "MetricDelta",
    "MetricResult",
    "MilvusVectorStore",
    "MinimaxLLM",
    "MongoDBAtlasVectorStore",
    "MongoDBLoader",
    "MoonshotLLM",
    "MultiStepRetriever",
    "MultimodalMessage",
    "NewsTool",
    "Node",
    "NodeFn",
    "NovitaLLM",
    "OTelExporter",
    "ObservationEvent",
    "ObsidianLoader",
    "OneDriveLoader",
    "OpenRouterLLM",
    "PDFLoader",
    "PDFReaderTool",
    "PGVectorStore",
    "PIIDetector",
    "PIIRedactor",
    "ParentDocumentRetriever",
    "ParquetLoader",
    "PerplexityLLM",
    # Memory / observability
    "PersistentAgentMemory",
    "PineconeVectorStore",
    # Plugins
    "PluginRegistry",
    "PostgresCheckpointer",
    "PowerPointLoader",
    # Prompts
    "PromptHub",
    "PromptTemplate",
    "PubMedSearchTool",
    "PydanticParser",
    "PythonREPLTool",
    "QdrantVectorStore",
    "QueryDecompositionRetriever",
    "RAGConfig",
    "RAGFusionRetriever",
    # Pipeline
    "RAGPipeline",
    "RSSLoader",
    "RTFLoader",
    "ReActAgent",
    "RecursionDepthError",
    "RecursiveCharacterTextSplitter",
    "RedactionResult",
    "RedisCheckpointer",
    "RedisConversationMemory",
    "RedisLoader",
    "RegexTool",
    "RegressionReport",
    "RelevancyMetric",
    # Retrieval
    "Retriever",
    "RouterModelSpec",
    "S3Loader",
    "SQLLoader",
    "SQLQueryTool",
    "SQLSchemaInspectionTool",
    "SQLiteCheckpointer",
    "SQLiteConversationMemory",
    "SQLiteVecStore",
    "SalesforceLoader",
    "SambaNovaLLM",
    "SelfQueryRetriever",
    "SelfRAGRetriever",
    "SemanticSplitter",
    "SentenceTextSplitter",
    "SentenceWindowRetriever",
    "SentenceWindowSplitter",
    "SentimentAnalysisTool",
    "ShellTool",
    "SimpleAgent",
    "SitemapLoader",
    "SitemapLoader",
    "SlackTool",
    "Span",
    "SpeechToTextTool",
    "StateField",
    "StateGraph",
    "StepBackRetriever",
    "StepEvent",
    "StringLoader",
    "StripeTool",
    "SummarizationTool",
    "SummaryBufferMemory",
    "SupabaseLoader",
    "SupervisorAgent",
    # Embeddings
    "SynapsekitEmbeddings",
    "TSVLoader",
    "Task",
    "TaskState",
    "TavilySearchTool",
    "TeamsLoader",
    "TextLoader",
    "TextToSpeechTool",
    "ThoughtEvent",
    "TogetherLLM",
    "TokenAwareSplitter",
    "TokenBufferMemory",
    "TokenEvent",
    "TokenTracer",
    "ToolRegistry",
    "ToolResult",
    "TopicRestrictor",
    "TraceEntry",
    "TraceSpan",
    "TracingMiddleware",
    "TracingUI",
    "TranslationTool",
    "TwilioTool",
    "TypedState",
    "VectorSearchTool",
    # Vector stores
    "VectorStore",
    "VertexAILLM",
    "VideoLoader",
    "WeatherTool",
    "WeaviateVectorStore",
    "WebLoader",
    "WebScraperTool",
    "WebSearchTool",
    "WikipediaLoader",
    "WikipediaTool",
    "WolframAlphaTool",
    "WorkerAgent",
    "WriterLLM",
    "XMLLoader",
    "XaiLLM",
    "YAMLLoader",
    "YouTubeLoader",
    "YouTubeSearchTool",
    "ZhipuLLM",
    "agent",
    "agent_node",
    "approval_node",
    # API stability markers
    "deprecated",
    "dynamic_route_node",
    "eval_case",
    "experimental",
    "fan_out_node",
    # Structured output
    "generate_structured",
    "get_mermaid_with_trace",
    "llm_node",
    "public_api",
    "rag_node",
    "sse_stream",
    "subgraph_node",
    # Tool decorator
    "tool",
    "ws_stream",
]

# Lazy imports for optional backends
_LAZY_IMPORTS = {
    # Vector stores
    "ChromaVectorStore": "retrieval.chroma",
    "FAISSVectorStore": "retrieval.faiss",
    "LanceDBVectorStore": "retrieval.lancedb",
    "MilvusVectorStore": "retrieval.milvus",
    "PGVectorStore": "retrieval.pgvector",
    "QdrantVectorStore": "retrieval.qdrant",
    "PineconeVectorStore": "retrieval.pinecone",
    "WeaviateVectorStore": "retrieval.weaviate",
    "SQLiteVecStore": "retrieval.sqlite_vec",
    # LLM providers
    "AsyncLRUCache": "llm._cache",
    "DynamoDBCacheBackend": "llm._cache_dynamodb",
    "MemcachedCacheBackend": "llm._cache_memcached",
    "AzureOpenAILLM": "llm.azure_openai",
    "CerebrasLLM": "llm.cerebras",
    "VertexAILLM": "llm.vertex_ai",
    "DeepSeekLLM": "llm.deepseek",
    "FireworksLLM": "llm.fireworks",
    "GroqLLM": "llm.groq",
    "GPT4AllLLM": "llm.gpt4all",
    "HuggingFaceLLM": "llm.huggingface",
    "OpenRouterLLM": "llm.openrouter",
    "PerplexityLLM": "llm.perplexity",
    "SambaNovaLLM": "llm.sambanova",
    "TogetherLLM": "llm.together",
    "MinimaxLLM": "llm.minimax",
    "NovitaLLM": "llm.novita",
    "MoonshotLLM": "llm.moonshot",
    "AlephAlphaLLM": "llm.aleph_alpha",
    "XaiLLM": "llm.xai",
    "WriterLLM": "llm.writer",
    "ZhipuLLM": "llm.zhipu",
    "LMStudioLLM": "llm.lmstudio",
    "VLLMLLM": "llm.vllm",
    "CloudflareLLM": "llm.cloudflare",
    # Checkpointers
    "RedisCheckpointer": "graph.checkpointers.redis",
    "PostgresCheckpointer": "graph.checkpointers.postgres",
    # Loaders
    "AirtableLoader": "loaders.airtable",
    "AudioLoader": "loaders.audio",
    "VideoLoader": "loaders.video",
    "DocxLoader": "loaders.docx",
    "ExcelLoader": "loaders.excel",
    "PowerPointLoader": "loaders.pptx",
    "TeamsLoader": "loaders.teams",
    "YAMLLoader": "loaders.yaml_loader",
    "DiscordLoader": "loaders.discord",
    "XMLLoader": "loaders.xml_loader",
    "GoogleDriveLoader": "loaders.google_drive",
    "HubSpotLoader": "loaders.hubspot",
    "MongoDBLoader": "loaders.mongodb",
    "ObsidianLoader": "loaders.obsidian",
    "OneDriveLoader": "loaders.onedrive",
    "AzureBlobLoader": "loaders.azure_blob",
    "BigQueryLoader": "loaders.bigquery",
    "S3Loader": "loaders.s3",
    "SalesforceLoader": "loaders.salesforce",
    "DropboxLoader": "loaders.dropbox",
    "ParquetLoader": "loaders.parquet",
    "RedisLoader": "loaders.redis_loader",
    "ElasticsearchLoader": "loaders.elasticsearch",
    "SitemapLoader": "loaders.sitemap",
    "YouTubeLoader": "loaders.youtube",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

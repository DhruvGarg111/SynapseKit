from .backend import SynapsekitEmbeddings
from .multimodal import (
    BaseMultimodalEmbeddings,
    ImageTextEmbeddings,
    MultimodalEmbeddings,
)

__all__ = [
    "ONNXEmbeddings",
    "SynapsekitEmbeddings",
    "BaseMultimodalEmbeddings",
    "ImageTextEmbeddings",
    "MultimodalEmbeddings",
    "ColPaliEmbeddings",
    "ColQwenEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "HuggingFaceEmbeddings",
    "JinaEmbeddings",
    "MistralEmbeddings",
    "MixedbreadEmbeddings",
    "NomicEmbeddings",
    "OpenAIEmbeddings",
    "VoyageEmbeddings",
]

_BACKENDS = {
    "ONNXEmbeddings": ".onnx",
    "OpenAIEmbeddings": ".openai",
    "CohereEmbeddings": ".cohere",
    "VoyageEmbeddings": ".voyage",
    "JinaEmbeddings": ".jina",
    "GeminiEmbeddings": ".gemini",
    "MistralEmbeddings": ".mistral",
    "NomicEmbeddings": ".nomic",
    "MixedbreadEmbeddings": ".mixedbread",
    "HuggingFaceEmbeddings": ".huggingface",
    "ColPaliEmbeddings": ".colpali",
    "ColQwenEmbeddings": ".colpali",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        import importlib

        mod = importlib.import_module(_BACKENDS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

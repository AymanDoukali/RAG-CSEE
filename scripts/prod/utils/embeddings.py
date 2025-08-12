from logging_config import logger
from langchain.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_id:int = 0) -> HuggingFaceEmbeddings:
    """
    Get a HuggingFace embedding model.

    Args:
        model_id (int): The ID of the model to retrieve.

            0: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    Returns:
        HuggingFaceEmbeddings: The requested embedding model.
    """
    model_kwargs = {}
    encode_kwargs = {}
    if model_id == 0:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        encode_kwargs = {"normalize_embeddings": True}

    elif model_id == 1:
        model_name = "dangvantuan/french-document-embedding"
        model_kwargs = {"device": "cpu",
                        "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs if model_kwargs else {}, encode_kwargs=encode_kwargs if encode_kwargs else {})
    logger.info(f"Using embedding model: {model_name}")
    return model
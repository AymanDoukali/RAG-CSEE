import os
from typing import List, Dict, Any
from logging_config import logger
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from embeddings import get_embedding_model


def save_dict_list_as_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save a list of dictionaries to a JSON file.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries to save
        file_path (str): Path to the output JSON file

    Raises:
        Exception: If saving fails
    """
    try:
        logger.info(f"📂 Saving {len(data)} dictionaries to JSON file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"❌ Error saving JSON file {file_path}: {str(e)}")
        raise


def load_json_as_dict_list(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file and return it as a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries loaded from JSON
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
        ValueError: If the JSON doesn't contain a list
    """
    try:
        logger.info(f"📂 Loading JSON file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the loaded data is a list
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list, got {type(data).__name__}")
        
        # Ensure all items in the list are dictionaries
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary, got {type(item).__name__}")
        
        logger.info(f"✅ Successfully loaded {len(data)} dictionaries from {file_path}")
        return data
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format in {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading JSON file {file_path}: {str(e)}")
        raise


def dict_list_to_text(data: list, save_path: str = None) -> str:
    """
    Concatenate the 'text' field from a list of dictionaries into a single string.
    Optionally save the result to a file.

    Args:
        data (list): List of dictionaries, each containing a 'text' key.
        save_path (str, optional): Path to save the concatenated text. If None, does not save.

    Returns:
        str: Concatenated text from all dictionaries.
    """
    all_text = "\n\n".join(item.get("text", "") for item in data)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        logger.info(f"✅ Saved concatenated text to {save_path}")
    return all_text


def load_faiss_index(persist_path: str, embeddings: HuggingFaceEmbeddings = get_embedding_model()) -> FAISS:
    """
    Load a FAISS index from a local directory.

    Args:
        persist_path (str): Path to the FAISS index directory.
        embeddings (HuggingFaceEmbeddings): Embedding model to use.

    Returns:
        FAISS: The loaded FAISS index.
    """
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"Index path {persist_path} not found.")
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def load_vectorstore(
    faiss_path: str,
    embeddings: HuggingFaceEmbeddings
) -> tuple:
    """
    Load embeddings and metadata from a FAISS vectorstore.

    Args:
        faiss_path (str): Path to the FAISS index directory.
        embeddings (Any): Embedding function or model used for loading.

    Returns:
        tuple: (all_embeddings, metadatas)
            all_embeddings: Embedding vectors from the FAISS index.
            metadatas: List of metadata dictionaries for each document.

    Raises:
        Exception: If loading the FAISS index fails.
    """
    try:
        vectorstore = load_faiss_index(faiss_path, embeddings)
        documents = list(vectorstore.docstore._dict.values())
        metadatas = [doc.metadata for doc in documents]

        all_embeddings = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

    except Exception as e:
        logger.error(f"❌ Error loading FAISS index from {faiss_path}: {str(e)}")
        raise
    return vectorstore, documents, all_embeddings, metadatas
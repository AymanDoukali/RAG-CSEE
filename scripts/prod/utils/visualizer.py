import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def visualize_2D(all_embeddings, documents):
    """
    Visualizes 2D embeddings using t-SNE and colors points by their associated document.
    This function reduces high-dimensional embeddings to 2D using t-SNE, then creates an interactive scatter plot
    where each point represents a chunk of text from a document. Hovering over a point displays metadata such as chunk ID, document name, page number, number of words, and a text preview.
    
    Args:
        all_embeddings (np.ndarray): Array of shape (n_samples, n_features) containing the embeddings to visualize.
        documents (List[Document]): List of document objects, each with a `metadata` dictionary containing keys:
            - 'doc_name': Name of the document.
            - 'chunk_id': Identifier for the chunk.
            - 'page_number': Page number in the document.
            - 'num_words': Number of words in the chunk.
            - 'page_content': Text content of the chunk.

    Returns:
        None: Displays an interactive Plotly figure.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(all_embeddings)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, opacity=0.8),
    text=[
        f"c_id: {doc.metadata.get('chunk_id', '')}\t<br>"
        f"doc_name: {doc.metadata.get('doc_name', '')}\t"
        f"page: {doc.metadata.get('page_number', '')}\t"
        f"n_words: {doc.metadata.get('num_words', 0)}<br>"
        f"text: {doc.page_content[:150]}..."
        for doc in documents
    ],
     hoverinfo='text'
    )])

    fig.update_layout(
        title='2D FAISS Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()


def visualize_2D_colored(all_embeddings, documents):
    """
    Visualizes 2D embeddings using t-SNE and colors points by their associated document.
    This function reduces high-dimensional embeddings to 2D using t-SNE, then creates an interactive scatter plot
    where each point represents a chunk of text from a document. Points are colored according to their document name,
    and hovering over a point displays metadata such as chunk ID, document name, page number, number of words, and a text preview.
   
    Args:
        all_embeddings (np.ndarray): Array of shape (n_samples, n_features) containing the embeddings to visualize.
        documents (List[Document]): List of document objects, each with a `metadata` dictionary containing keys:
            - 'doc_name': Name of the document.
            - 'chunk_id': Identifier for the chunk.
            - 'page_number': Page number in the document.
            - 'num_words': Number of words in the chunk.
            - 'page_content': Text content of the chunk.
            
    Returns:
        None: Displays an interactive Plotly figure.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(all_embeddings)

    # Extract unique document names for color mapping
    doc_names = [doc.metadata.get('doc_name', '') for doc in documents]
    unique_doc_names = sorted(set(doc_names))
    
    # Create color mapping
    color_map = {}
    for i, name in enumerate(unique_doc_names):
        color_map[name] = i
    
    # Create color array based on document names
    colors = [color_map[name] for name in doc_names]

    # Create the 2D scatter plot with colored dots by document
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(
            size=5, 
            opacity=0.8,
            color=colors,  # Use colors array based on document names
            colorscale='Viridis',  # Choose a colorscale
            colorbar=dict(
                title="Documents",
                tickvals=[color_map[name] for name in unique_doc_names],
                ticktext=unique_doc_names
            )
        ),
        text=[
            f"c_id: {doc.metadata.get('chunk_id', '')}\t<br>"
            f"doc_name: {doc.metadata.get('doc_name', -1)}\t"
            f"page: {doc.metadata.get('page_number', '')}\t"
            f"n_words: {doc.metadata.get('num_words', -1)}<br>"
            f"text: {doc.page_content[:150]}..."
            for doc in documents
        ],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D FAISS Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()
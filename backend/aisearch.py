import json
import logging
import os
import subprocess
import time
from typing import List

from fastapi import UploadFile
from dotenv import load_dotenv

# Azure Search imports
try:
    from azure.core.exceptions import ResourceExistsError
    from azure.identity import DefaultAzureCredential
    from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
    from azure.search.documents.indexes.models import (
        AzureOpenAIEmbeddingSkill,
        AzureOpenAIVectorizerParameters,
        AzureOpenAIVectorizer,
        FieldMapping,
        HnswAlgorithmConfiguration,
        HnswParameters,
        IndexProjectionMode,
        InputFieldMappingEntry,
        OutputFieldMappingEntry,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchIndexer,
        SearchIndexerDataContainer,
        SearchIndexerDataSourceConnection,
        SearchIndexerDataSourceType,
        SearchIndexerDataUserAssignedIdentity,
        SearchIndexerIndexProjection,
        SearchIndexerIndexProjectionSelector,
        SearchIndexerIndexProjectionsParameters,
        SearchIndexerSkillset,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        SplitSkill,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )
    from azure.storage.blob import BlobServiceClient
    azure_available = True
except ImportError:
    azure_available = False

# OpenSearch imports
try:
    from opensearchpy import OpenSearch
    opensearch_available = True
except ImportError:
    opensearch_available = False

# FAISS imports
try:
    import faiss
    import numpy as np
    import pickle
    local_available = True
except ImportError:
    local_available = False

load_dotenv("../.env", override=True)

SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "azure").lower()
EMBEDDINGS_DIMENSIONS = 3072

def setup_index_backend(index_name: str):
    if SEARCH_BACKEND == "azure" and azure_available:
        setup_azure_index(index_name)
    elif SEARCH_BACKEND == "opensearch" and opensearch_available:
        setup_opensearch_index(index_name)
    elif SEARCH_BACKEND == "local" and local_available:
        setup_local_index(index_name)
    else:
        raise RuntimeError(f"Unsupported or unavailable search backend: {SEARCH_BACKEND}")

def setup_azure_index(index_name: str):
    logger = logging.getLogger("setup_index")
    logger.setLevel(logging.INFO)
    azure_credential = DefaultAzureCredential()
    AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    AZURE_OPENAI_EMBEDDING_MODEL = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    UAMI_RESOURCE_ID = os.environ["UAMI_RESOURCE_ID"]
    AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    AZURE_STORAGE_ENDPOINT = os.getenv("AZURE_STORAGE_ACCOUNT_ENDPOINT")
    AZURE_STORAGE_CONNECTION_STRING = f"ResourceId={os.getenv('AZURE_STORAGE_ACCOUNT_ID')}"
    azure_storage_container = index_name

    index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, azure_credential)
    indexer_client = SearchIndexerClient(AZURE_SEARCH_ENDPOINT, azure_credential)
    blob_client = BlobServiceClient(account_url=AZURE_STORAGE_ENDPOINT, credential=azure_credential)
    container_client = blob_client.get_container_client(azure_storage_container)
    if not container_client.exists():
        container_client.create_container()

    if index_name not in [ds.name for ds in indexer_client.get_data_source_connections()]:
        indexer_client.create_data_source_connection(
            SearchIndexerDataSourceConnection(
                name=index_name,
                type=SearchIndexerDataSourceType.AZURE_BLOB,
                connection_string=AZURE_STORAGE_CONNECTION_STRING,
                identity=SearchIndexerDataUserAssignedIdentity(resource_id=UAMI_RESOURCE_ID),
                container=SearchIndexerDataContainer(name=azure_storage_container))
        )

    if index_name not in [i.name for i in index_client.list_indexes()]:
        index_client.create_index(
            SearchIndex(
                name=index_name,
                fields=[
                    SearchableField(name="chunk_id", key=True, analyzer_name="keyword", sortable=True),
                    SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
                    SearchableField(name="title"),
                    SearchableField(name="chunk"),
                    SearchField(
                        name="text_vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=EMBEDDINGS_DIMENSIONS,
                        vector_search_profile_name="vp",
                        stored=True,
                        hidden=False)
                ],
                vector_search=VectorSearch(
                    algorithms=[
                        HnswAlgorithmConfiguration(
                            name="algo",
                            parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE))
                    ],
                    vectorizers=[
                        AzureOpenAIVectorizer(
                            vectorizer_name="openai_vectorizer",
                            parameters=AzureOpenAIVectorizerParameters(
                                resource_url=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                                auth_identity=SearchIndexerDataUserAssignedIdentity(resource_id=UAMI_RESOURCE_ID),
                                deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                                model_name=AZURE_OPENAI_EMBEDDING_MODEL
                            )
                        )
                    ],
                    profiles=[
                        VectorSearchProfile(name="vp", algorithm_configuration_name="algo", vectorizer_name="openai_vectorizer")
                    ]
                ),
                semantic_search=SemanticSearch(
                    configurations=[
                        SemanticConfiguration(
                            name="default",
                            prioritized_fields=SemanticPrioritizedFields(
                                title_field=SemanticField(field_name="title"),
                                content_fields=[SemanticField(field_name="chunk")]
                            )
                        )
                    ],
                    default_configuration_name="default"
                )
            )
        )

def setup_opensearch_index(index_name: str):
    logger = logging.getLogger("opensearch")
    logger.setLevel(logging.INFO)
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    index = index_name
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
        use_ssl=False
    )

    if client.indices.exists(index=index):
        logger.info(f"OpenSearch index {index} already exists.")
    else:
        logger.info(f"Creating OpenSearch index: {index}")
        settings = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "chunk": {"type": "text"},
                    "text_vector": {"type": "knn_vector", "dimension": EMBEDDINGS_DIMENSIONS}
                }
            }
        }
        client.indices.create(index=index, body=settings)

def setup_local_index(index_name: str):
    logger = logging.getLogger("faiss")
    logger.setLevel(logging.INFO)
    index_path = f"./data/faiss_indexes/{index_name}.index"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if os.path.exists(index_path):
        logger.info(f"FAISS index {index_name} already exists at {index_path}.")
    else:
        logger.info(f"Creating FAISS index: {index_name}")
        index = faiss.IndexFlatL2(EMBEDDINGS_DIMENSIONS)
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("index-bootstrap")

    folders = [f for f in os.listdir("./data/ai-search-index") if os.path.isdir(os.path.join("./data/ai-search-index", f))]
    for index_name in folders:
        logger.info(f"Setting up index: {index_name}")
        setup_index_backend(index_name)

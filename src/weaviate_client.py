"""
Helper kết nối Weaviate — dùng chung giữa Task 4 (indexing) và Task 5
(semantic search), tránh trùng lặp logic kết nối.

Mặc định kết nối Weaviate local (Docker) cho môi trường dev. Khi
WEAVIATE_URL được set trong .env (vd. khi deploy backend lên môi trường
không chạy được Docker như Hugging Face Spaces), chuyển sang kết nối
Weaviate Cloud bằng WEAVIATE_URL + WEAVIATE_API_KEY.
"""

import os

import weaviate
from weaviate.classes.init import Auth


def connect_weaviate() -> weaviate.WeaviateClient:
    weaviate_url = os.getenv("WEAVIATE_URL", "")
    if weaviate_url:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY", "")),
        )
    return weaviate.connect_to_local()

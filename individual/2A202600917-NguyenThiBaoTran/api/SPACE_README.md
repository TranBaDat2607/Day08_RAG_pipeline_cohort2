---
title: LuatMaTuy AI Chat API
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

# LuậtMaTuý AI — Chat API

Backend FastAPI bọc pipeline RAG (Task 9 hybrid retrieval + Task 10 generation
có citation) cho demo chatbot. Endpoint chính: `POST /chat` nhận
`{"query": "..."}`, trả về `{"answer", "sources", "retrieval_source"}`.

Repo nguồn: xem GitHub repo của dự án — Space này được đồng bộ tự động qua
GitHub Actions (`.github/workflows/deploy-backend.yml`).

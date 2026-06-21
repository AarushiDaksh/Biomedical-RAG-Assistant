import json
from fastapi.testclient import TestClient


def test_documents_lists_indexed(tmp_path, monkeypatch):
    import rag.config as config
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))

    docs = tmp_path / "documents.jsonl"
    docs.write_text(json.dumps({
        "document_id": "d1", "title": "Paper One", "file_name": "one.pdf",
        "pages": [{"page_number": 1, "text": "x"}], "ocr_used": False,
    }) + "\n", encoding="utf-8")

    import backend.main as m
    client = TestClient(m.app)

    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["document_id"] == "d1"
    assert body[0]["pages"] == 1
    assert body[0]["scanned"] is False


def test_documents_empty_when_no_file(tmp_path, monkeypatch):
    import rag.config as config
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))  # empty dir, no jsonl
    import backend.main as m
    client = TestClient(m.app)
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == []

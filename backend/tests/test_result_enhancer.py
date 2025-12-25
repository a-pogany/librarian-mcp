import pytest

from core.result_enhancer import ResultEnhancer


def test_enhance_email_result_strips_metadata_by_default():
    enhancer = ResultEnhancer(summary_length=80)
    result = {
        "id": "emails/2024/message.eml",
        "file_path": "emails/2024/message.eml",
        "file_type": ".eml",
        "snippet": "project deadline",
        "relevance_score": 0.9,
        "content_preview": "Hello team, the project deadline has moved to Friday.",
        "metadata": {
            "doc_type": "email",
            "from": "sender@example.com",
            "to": ["team@example.com"],
            "cc": ["lead@example.com"],
            "date": "2024-01-15T10:30:00",
            "subject": "Re: Project Timeline",
            "has_attachments": True,
            "attachment_count": 1,
            "attachments": [{"filename": "plan.pdf"}],
            "thread_id": "thread-123"
        }
    }

    enhanced = enhancer.enhance([result])[0]
    assert enhanced["type"] == "email"
    assert enhanced["title"] == "Re: Project Timeline"
    assert enhanced["from"] == "sender@example.com"
    assert enhanced["to"] == ["team@example.com"]
    assert enhanced["has_attachments"] is True
    assert "metadata" not in enhanced
    assert "content_preview" not in enhanced


def test_enhance_email_includes_metadata_when_requested():
    enhancer = ResultEnhancer()
    result = {
        "id": "emails/2024/message.eml",
        "file_path": "emails/2024/message.eml",
        "file_type": ".eml",
        "content_preview": "Body preview",
        "metadata": {"doc_type": "email", "subject": "Hello"}
    }

    enhanced = enhancer.enhance([result], include_full_metadata=True)[0]
    assert enhanced["metadata"]["subject"] == "Hello"
    assert enhanced["content_preview"] == "Body preview"


def test_enhance_document_sets_title_and_headings():
    enhancer = ResultEnhancer(summary_length=40)
    result = {
        "id": "docs/guide.md",
        "file_path": "docs/guide.md",
        "file_type": ".md",
        "file_name": "guide.md",
        "headings": ["Getting Started", "Install", "Config", "Usage", "FAQ", "Extra"],
        "content_preview": "This guide explains how to set things up quickly.",
        "doc_type": "guide",
        "tags": ["setup", "quickstart"],
        "last_modified": "2024-01-10T09:00:00"
    }

    enhanced = enhancer.enhance([result])[0]
    assert enhanced["type"] == "document"
    assert enhanced["title"] == "Getting Started"
    assert enhanced["headings"] == ["Getting Started", "Install", "Config", "Usage", "FAQ"]
    assert enhanced["doc_type"] == "guide"
    assert "metadata" not in enhanced

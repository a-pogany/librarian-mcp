"""
Integration tests for email search operators feature.

Tests the full pipeline:
- Query parsing in search_emails()
- Filter application with parsed operators
- Integration with different search modes
- Folder extraction from file paths
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.core.query_parser import QueryOperatorParser, ParsedQuery


class TestSearchEmailsWithOperators:
    """Integration tests for search_emails() with operator parsing"""

    @pytest.fixture
    def mock_search_engine(self):
        """Create a mock hybrid search engine"""
        engine = MagicMock()
        engine.search.return_value = [
            {
                'doc_id': 'test/emails/inbox/email1.eml',
                'content': 'Meeting tomorrow with the team',
                'metadata': {
                    'from': 'alice@example.com',
                    'to': ['bob@example.com'],
                    'cc': [],
                    'subject': 'Team Meeting',
                    'folder': 'inbox',
                    'has_attachments': False,
                    'date': '2024-06-15'
                },
                'score': 0.95
            },
            {
                'doc_id': 'test/emails/sent/email2.eml',
                'content': 'Sent budget report',
                'metadata': {
                    'from': 'bob@example.com',
                    'to': ['alice@example.com', 'charlie@example.com'],
                    'cc': ['manager@example.com'],
                    'subject': 'Budget Report Q4',
                    'folder': 'sent',
                    'has_attachments': True,
                    'date': '2024-06-10'
                },
                'score': 0.85
            },
            {
                'doc_id': 'test/emails/important/email3.eml',
                'content': 'Project deadline reminder',
                'metadata': {
                    'from': 'kiraly@example.com',
                    'to': ['team@example.com'],
                    'cc': [],
                    'subject': 'Project Deadline',
                    'folder': 'important',
                    'has_attachments': False,
                    'date': '2024-06-20'
                },
                'score': 0.75
            }
        ]
        return engine

    def test_operator_parsing_extracts_from(self):
        """Test that from: operator is correctly parsed"""
        parser = QueryOperatorParser()
        result = parser.parse("from:kiraly project deadline")

        assert result.free_text == "project deadline"
        assert result.operators.get("from") == "kiraly"

        params = result.get_filter_params()
        assert params.get("sender") == "kiraly"

    def test_operator_parsing_extracts_to(self):
        """Test that to: operator is correctly parsed"""
        parser = QueryOperatorParser()
        result = parser.parse("to:alice meeting notes")

        assert result.free_text == "meeting notes"
        assert result.operators.get("to") == "alice"

        params = result.get_filter_params()
        assert params.get("recipient") == "alice"

    def test_operator_parsing_extracts_folder(self):
        """Test that in: (folder) operator is correctly parsed"""
        parser = QueryOperatorParser()
        result = parser.parse("in:important urgent emails")

        assert result.free_text == "urgent emails"
        assert result.operators.get("in") == "important"

        params = result.get_filter_params()
        assert params.get("folder") == "important"

    def test_combined_operators(self):
        """Test multiple operators in one query"""
        parser = QueryOperatorParser()
        result = parser.parse("from:kiraly in:important project status")

        assert result.free_text == "project status"
        assert result.operators.get("from") == "kiraly"
        assert result.operators.get("in") == "important"

        params = result.get_filter_params()
        assert params.get("sender") == "kiraly"
        assert params.get("folder") == "important"

    def test_quoted_operator_values(self):
        """Test operators with quoted values containing spaces"""
        parser = QueryOperatorParser()
        result = parser.parse('from:"John Doe" subject:"Q4 Report" meeting')

        assert result.free_text == "meeting"
        assert result.operators.get("from") == "John Doe"
        assert result.operators.get("subject") == "Q4 Report"

    def test_date_operators(self):
        """Test date range operators"""
        parser = QueryOperatorParser()
        result = parser.parse("after:2024-01-01 before:2024-12-31 quarterly report")

        params = result.get_filter_params()
        assert params.get("date_after") == "2024-01-01"
        assert params.get("date_before") == "2024-12-31"

    def test_has_attachment_operator(self):
        """Test has:attachment operator"""
        parser = QueryOperatorParser()
        result = parser.parse("has:attachment invoice")

        params = result.get_filter_params()
        assert params.get("has_attachments") is True

    def test_has_attachments_plural(self):
        """Test has:attachments (plural) operator"""
        parser = QueryOperatorParser()
        result = parser.parse("has:attachments report")

        params = result.get_filter_params()
        assert params.get("has_attachments") is True

    def test_thread_operator(self):
        """Test thread: operator for thread filtering"""
        parser = QueryOperatorParser()
        result = parser.parse("thread:abc123xyz replies")

        params = result.get_filter_params()
        assert params.get("thread_id") == "abc123xyz"

    def test_cc_operator(self):
        """Test cc: operator"""
        parser = QueryOperatorParser()
        result = parser.parse("cc:manager project update")

        params = result.get_filter_params()
        assert params.get("cc") == "manager"


class TestApplyEmailFilters:
    """Test the _apply_email_filters function with new parameters"""

    @pytest.fixture
    def sample_results(self):
        """Sample search results for filter testing"""
        return [
            {
                'doc_id': 'email1.eml',
                'metadata': {
                    'from': 'alice@example.com',
                    'to': ['bob@example.com', 'carol@example.com'],
                    'cc': ['manager@example.com'],
                    'folder': 'inbox',
                    'has_attachments': True
                }
            },
            {
                'doc_id': 'email2.eml',
                'metadata': {
                    'from': 'kiraly@example.com',
                    'to': ['team@example.com'],
                    'cc': [],
                    'folder': 'important',
                    'has_attachments': False
                }
            },
            {
                'doc_id': 'email3.eml',
                'metadata': {
                    'from': 'bob@example.com',
                    'to': ['alice@example.com'],
                    'cc': ['cfo@example.com'],
                    'folder': 'sent',
                    'has_attachments': True
                }
            }
        ]

    def test_filter_by_recipient(self, sample_results):
        """Test filtering by recipient (to:)"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(sample_results, recipient="bob")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email1.eml'

    def test_filter_by_recipient_partial_match(self, sample_results):
        """Test recipient filter with partial match"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(sample_results, recipient="team")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email2.eml'

    def test_filter_by_cc(self, sample_results):
        """Test filtering by CC recipient"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(sample_results, cc="manager")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email1.eml'

    def test_filter_by_folder(self, sample_results):
        """Test filtering by folder (in:)"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(sample_results, folder="important")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email2.eml'

    def test_filter_by_folder_case_insensitive(self, sample_results):
        """Test folder filter is case-insensitive"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(sample_results, folder="IMPORTANT")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email2.eml'

    def test_combined_filters(self, sample_results):
        """Test combining multiple filters"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(
            sample_results,
            sender="alice",
            has_attachments=True
        )

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email1.eml'

    def test_combined_filters_no_match(self, sample_results):
        """Test combined filters with no matching results"""
        from backend.mcp_server.tools import _apply_email_filters

        filtered = _apply_email_filters(
            sample_results,
            sender="kiraly",
            folder="inbox"  # kiraly's emails are in 'important', not 'inbox'
        )

        assert len(filtered) == 0

    def test_filter_with_missing_metadata(self):
        """Test filter handles missing metadata gracefully"""
        from backend.mcp_server.tools import _apply_email_filters

        results = [
            {
                'doc_id': 'email1.eml',
                'metadata': {}  # No metadata
            },
            {
                'doc_id': 'email2.eml',
                'metadata': {
                    'from': 'alice@example.com',
                    'folder': 'inbox'
                }
            }
        ]

        # Should not raise exception
        filtered = _apply_email_filters(results, folder="inbox")

        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 'email2.eml'


class TestFolderExtraction:
    """Test folder extraction from file paths in EMLParser"""

    def test_inbox_folder_extraction(self):
        """Test extracting 'inbox' folder from path"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        folder = parser._infer_folder("/docs/emails/inbox/email.eml")

        assert folder == "inbox"

    def test_sent_folder_extraction(self):
        """Test extracting 'sent' folder from path"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        folder = parser._infer_folder("/docs/emails/sent/email.eml")

        assert folder == "sent"

    def test_important_folder_extraction(self):
        """Test extracting 'important' folder from path"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        folder = parser._infer_folder("/docs/emails/important/email.eml")

        assert folder == "important"

    def test_nested_folder_extraction(self):
        """Test folder extraction from nested path"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        folder = parser._infer_folder("/data/backup/emails/projects/2024/email.eml")

        assert folder == "projects"

    def test_sent_items_folder(self):
        """Test 'Sent Items' folder extraction"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        folder = parser._infer_folder("/docs/Sent Items/email.eml")

        assert folder == "sent items"

    def test_unknown_folder_uses_parent(self):
        """Test that unknown folder uses parent directory name as fallback"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        # When no known folder is found, uses parent dir name
        folder = parser._infer_folder("/random/custom_folder/email.eml")

        assert folder == "custom_folder"

    def test_root_folder_when_no_parent(self):
        """Test that 'root' is returned when no valid parent exists"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        # Parent is in skip list, so returns 'root'
        folder = parser._infer_folder("/docs/email.eml")

        assert folder == "root"

    def test_generic_email_folder_skipped(self):
        """Test that generic 'emails' folder is skipped"""
        from backend.core.email_parser import EMLParser

        parser = EMLParser()
        # 'emails' should be skipped, 'inbox' should be found
        folder = parser._infer_folder("/docs/emails/inbox/email.eml")

        assert folder == "inbox"


class TestSearchEmailsIntegration:
    """Full integration tests for search_emails with operator parsing"""

    def test_query_without_operators(self):
        """Test search with no operators returns full query as free text"""
        parser = QueryOperatorParser()
        result = parser.parse("project deadline meeting notes")

        assert result.free_text == "project deadline meeting notes"
        assert result.operators == {}
        assert not result.has_operators()

    def test_only_operators_no_text(self):
        """Test query with only operators and no free text"""
        parser = QueryOperatorParser()
        result = parser.parse("from:alice to:bob")

        assert result.free_text == ""
        assert result.operators == {"from": "alice", "to": "bob"}
        assert result.has_operators()

    def test_operators_preserve_original_query(self):
        """Test that original query is preserved"""
        parser = QueryOperatorParser()
        original = "from:alice in:important project"
        result = parser.parse(original)

        assert result.original_query == original

    def test_all_operators_combined(self):
        """Test using all operators together"""
        parser = QueryOperatorParser()
        result = parser.parse(
            "from:alice to:bob cc:charlie subject:budget "
            "in:important has:attachment after:2024-01-01 before:2024-12-31 "
            "thread:xyz project review"
        )

        assert result.free_text == "project review"
        params = result.get_filter_params()

        assert params["sender"] == "alice"
        assert params["recipient"] == "bob"
        assert params["cc"] == "charlie"
        assert params["subject_contains"] == "budget"
        assert params["folder"] == "important"
        assert params["has_attachments"] is True
        assert params["date_after"] == "2024-01-01"
        assert params["date_before"] == "2024-12-31"
        assert params["thread_id"] == "xyz"

    def test_unknown_operators_treated_as_text(self):
        """Test unknown operators are kept in free text"""
        parser = QueryOperatorParser()
        result = parser.parse("custom:value from:alice project")

        assert result.free_text == "custom:value project"
        assert result.operators == {"from": "alice"}

    def test_duplicate_operators_use_last_value(self):
        """Test duplicate operators use last value with warning"""
        parser = QueryOperatorParser()
        result = parser.parse("from:alice from:bob meeting")

        assert result.operators["from"] == "bob"
        assert len(result.parse_warnings) == 1
        assert "Duplicate" in result.parse_warnings[0]

    def test_case_insensitive_operators(self):
        """Test operators are case-insensitive"""
        parser = QueryOperatorParser()
        result = parser.parse("FROM:alice TO:bob SUBJECT:budget meeting")

        assert result.operators == {"from": "alice", "to": "bob", "subject": "budget"}


class TestEdgeCases:
    """Edge case tests for robustness"""

    def test_empty_query(self):
        """Test empty query handling"""
        parser = QueryOperatorParser()
        result = parser.parse("")

        assert result.free_text == ""
        assert result.operators == {}

    def test_whitespace_only_query(self):
        """Test whitespace-only query handling"""
        parser = QueryOperatorParser()
        result = parser.parse("   ")

        assert result.free_text == ""

    def test_operator_at_end_of_query(self):
        """Test operator at end of query"""
        parser = QueryOperatorParser()
        result = parser.parse("meeting notes from:alice")

        assert result.free_text == "meeting notes"
        assert result.operators["from"] == "alice"

    def test_multiple_spaces_normalized(self):
        """Test multiple spaces are normalized"""
        parser = QueryOperatorParser()
        result = parser.parse("from:alice    project     deadline")

        assert result.free_text == "project deadline"

    def test_quoted_value_with_colon(self):
        """Test quoted value containing colon"""
        parser = QueryOperatorParser()
        result = parser.parse('subject:"Meeting @ 3:00pm" reminder')

        assert result.operators["subject"] == "Meeting @ 3:00pm"
        assert result.free_text == "reminder"

    def test_operators_mixed_with_text(self):
        """Test operators interspersed with text"""
        parser = QueryOperatorParser()
        result = parser.parse("urgent from:alice meeting to:bob notes")

        assert result.free_text == "urgent meeting notes"
        assert result.operators == {"from": "alice", "to": "bob"}

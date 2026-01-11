"""
Unit tests for QueryOperatorParser.

Tests cover:
- Basic operator parsing
- Quoted values
- Multiple operators
- Edge cases (empty queries, only operators, unknown operators)
- Filter parameter conversion
"""

import pytest
from backend.core.query_parser import QueryOperatorParser, ParsedQuery


class TestQueryOperatorParser:
    """Test suite for QueryOperatorParser"""

    @pytest.fixture
    def parser(self):
        """Create parser instance for tests"""
        return QueryOperatorParser()

    # =========================================================================
    # Basic Operator Parsing
    # =========================================================================

    def test_parse_simple_from_operator(self, parser):
        """Test parsing single from: operator"""
        result = parser.parse("from:kiraly project deadline")
        assert result.free_text == "project deadline"
        assert result.operators == {"from": "kiraly"}
        assert result.operator_count == 1

    def test_parse_simple_to_operator(self, parser):
        """Test parsing single to: operator"""
        result = parser.parse("to:john meeting notes")
        assert result.free_text == "meeting notes"
        assert result.operators == {"to": "john"}

    def test_parse_subject_operator(self, parser):
        """Test parsing subject: operator"""
        result = parser.parse("subject:budget important")
        assert result.free_text == "important"
        assert result.operators == {"subject": "budget"}

    def test_parse_in_folder_operator(self, parser):
        """Test parsing in: (folder) operator"""
        result = parser.parse("in:important urgent emails")
        assert result.free_text == "urgent emails"
        assert result.operators == {"in": "important"}

    def test_parse_has_attachment_operator(self, parser):
        """Test parsing has:attachment operator"""
        result = parser.parse("has:attachment budget report")
        assert result.free_text == "budget report"
        assert result.operators == {"has": "attachment"}

    def test_parse_after_date_operator(self, parser):
        """Test parsing after: date operator"""
        result = parser.parse("after:2024-01-01 project update")
        assert result.free_text == "project update"
        assert result.operators == {"after": "2024-01-01"}

    def test_parse_before_date_operator(self, parser):
        """Test parsing before: date operator"""
        result = parser.parse("before:2024-12-31 year end")
        assert result.free_text == "year end"
        assert result.operators == {"before": "2024-12-31"}

    def test_parse_thread_operator(self, parser):
        """Test parsing thread: operator"""
        result = parser.parse("thread:abc123 discussion")
        assert result.free_text == "discussion"
        assert result.operators == {"thread": "abc123"}

    def test_parse_cc_operator(self, parser):
        """Test parsing cc: operator"""
        result = parser.parse("cc:team meeting")
        assert result.free_text == "meeting"
        assert result.operators == {"cc": "team"}

    # =========================================================================
    # Quoted Values
    # =========================================================================

    def test_parse_quoted_subject(self, parser):
        """Test parsing quoted value in subject: operator"""
        result = parser.parse('subject:"Q4 Report" urgent')
        assert result.free_text == "urgent"
        assert result.operators == {"subject": "Q4 Report"}

    def test_parse_quoted_from(self, parser):
        """Test parsing quoted value in from: operator"""
        result = parser.parse('from:"John Doe" meeting')
        assert result.free_text == "meeting"
        assert result.operators == {"from": "John Doe"}

    def test_parse_quoted_in_folder(self, parser):
        """Test parsing quoted folder name"""
        result = parser.parse('in:"Sent Items" response')
        assert result.free_text == "response"
        assert result.operators == {"in": "Sent Items"}

    def test_parse_quoted_with_special_chars(self, parser):
        """Test quoted value with special characters"""
        result = parser.parse('subject:"Meeting @ 3pm" reminder')
        assert result.free_text == "reminder"
        assert result.operators == {"subject": "Meeting @ 3pm"}

    # =========================================================================
    # Multiple Operators
    # =========================================================================

    def test_parse_multiple_operators(self, parser):
        """Test parsing multiple operators in one query"""
        result = parser.parse("from:alice to:bob meeting")
        assert result.free_text == "meeting"
        assert result.operators == {"from": "alice", "to": "bob"}
        assert result.operator_count == 2

    def test_parse_three_operators(self, parser):
        """Test parsing three operators"""
        result = parser.parse("from:alice to:bob subject:budget quarterly review")
        assert result.free_text == "quarterly review"
        assert result.operators == {"from": "alice", "to": "bob", "subject": "budget"}
        assert result.operator_count == 3

    def test_parse_all_operators(self, parser):
        """Test parsing query with many operators"""
        result = parser.parse(
            "from:alice to:bob cc:charlie subject:budget "
            "in:important has:attachment after:2024-01-01 before:2024-12-31 "
            "project review"
        )
        assert result.free_text == "project review"
        assert result.operators["from"] == "alice"
        assert result.operators["to"] == "bob"
        assert result.operators["cc"] == "charlie"
        assert result.operators["subject"] == "budget"
        assert result.operators["in"] == "important"
        assert result.operators["has"] == "attachment"
        assert result.operators["after"] == "2024-01-01"
        assert result.operators["before"] == "2024-12-31"

    def test_parse_operators_mixed_with_text(self, parser):
        """Test operators interspersed with free text"""
        result = parser.parse("urgent from:alice meeting to:bob notes")
        assert result.free_text == "urgent meeting notes"
        assert result.operators == {"from": "alice", "to": "bob"}

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_parse_empty_query(self, parser):
        """Test parsing empty query"""
        result = parser.parse("")
        assert result.free_text == ""
        assert result.operators == {}
        assert result.operator_count == 0

    def test_parse_whitespace_query(self, parser):
        """Test parsing whitespace-only query"""
        result = parser.parse("   ")
        assert result.free_text == ""
        assert result.operators == {}

    def test_parse_only_operators_no_text(self, parser):
        """Test parsing query with only operators, no free text"""
        result = parser.parse("from:kiraly to:john")
        assert result.free_text == ""
        assert result.operators == {"from": "kiraly", "to": "john"}
        assert result.operator_count == 2

    def test_parse_no_operators(self, parser):
        """Test parsing query without any operators"""
        result = parser.parse("project deadline meeting notes")
        assert result.free_text == "project deadline meeting notes"
        assert result.operators == {}
        assert result.operator_count == 0

    def test_parse_unknown_operator(self, parser):
        """Test that unknown operators are treated as text"""
        result = parser.parse("custom:value project")
        assert result.free_text == "custom:value project"
        assert result.operators == {}

    def test_parse_duplicate_operator(self, parser):
        """Test duplicate operators - last value wins"""
        result = parser.parse("from:alice from:bob meeting")
        assert result.free_text == "meeting"
        assert result.operators == {"from": "bob"}
        assert len(result.parse_warnings) == 1
        assert "Duplicate" in result.parse_warnings[0]

    def test_parse_case_insensitive_operators(self, parser):
        """Test that operator names are case-insensitive"""
        result = parser.parse("FROM:alice TO:bob meeting")
        assert result.free_text == "meeting"
        assert result.operators == {"from": "alice", "to": "bob"}

    def test_parse_preserves_original_query(self, parser):
        """Test that original query is preserved"""
        original = "from:alice project deadline"
        result = parser.parse(original)
        assert result.original_query == original

    def test_parse_normalizes_whitespace(self, parser):
        """Test that multiple spaces are normalized in free text"""
        result = parser.parse("from:alice   project    deadline")
        assert result.free_text == "project deadline"

    # =========================================================================
    # Filter Parameter Conversion
    # =========================================================================

    def test_get_filter_params_from(self, parser):
        """Test filter params conversion for from: operator"""
        result = parser.parse("from:kiraly project")
        params = result.get_filter_params()
        assert params == {"sender": "kiraly"}

    def test_get_filter_params_to(self, parser):
        """Test filter params conversion for to: operator"""
        result = parser.parse("to:john project")
        params = result.get_filter_params()
        assert params == {"recipient": "john"}

    def test_get_filter_params_cc(self, parser):
        """Test filter params conversion for cc: operator"""
        result = parser.parse("cc:team project")
        params = result.get_filter_params()
        assert params == {"cc": "team"}

    def test_get_filter_params_subject(self, parser):
        """Test filter params conversion for subject: operator"""
        result = parser.parse("subject:budget project")
        params = result.get_filter_params()
        assert params == {"subject_contains": "budget"}

    def test_get_filter_params_in_folder(self, parser):
        """Test filter params conversion for in: operator"""
        result = parser.parse("in:important project")
        params = result.get_filter_params()
        assert params == {"folder": "important"}

    def test_get_filter_params_thread(self, parser):
        """Test filter params conversion for thread: operator"""
        result = parser.parse("thread:abc123 project")
        params = result.get_filter_params()
        assert params == {"thread_id": "abc123"}

    def test_get_filter_params_dates(self, parser):
        """Test filter params conversion for date operators"""
        result = parser.parse("after:2024-01-01 before:2024-12-31 project")
        params = result.get_filter_params()
        assert params == {
            "date_after": "2024-01-01",
            "date_before": "2024-12-31"
        }

    def test_get_filter_params_has_attachment(self, parser):
        """Test filter params conversion for has:attachment"""
        result = parser.parse("has:attachment project")
        params = result.get_filter_params()
        assert params == {"has_attachments": True}

    def test_get_filter_params_has_attachments_plural(self, parser):
        """Test filter params conversion for has:attachments (plural)"""
        result = parser.parse("has:attachments project")
        params = result.get_filter_params()
        assert params == {"has_attachments": True}

    def test_get_filter_params_all(self, parser):
        """Test filter params conversion for all operators"""
        result = parser.parse(
            "from:alice to:bob cc:charlie subject:budget "
            "in:important has:attachment after:2024-01-01 before:2024-12-31 "
            "thread:xyz project"
        )
        params = result.get_filter_params()
        assert params == {
            "sender": "alice",
            "recipient": "bob",
            "cc": "charlie",
            "subject_contains": "budget",
            "folder": "important",
            "has_attachments": True,
            "date_after": "2024-01-01",
            "date_before": "2024-12-31",
            "thread_id": "xyz"
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def test_has_operators_true(self, parser):
        """Test has_operators returns True when operators present"""
        result = parser.parse("from:alice project")
        assert result.has_operators() is True

    def test_has_operators_false(self, parser):
        """Test has_operators returns False when no operators"""
        result = parser.parse("project deadline")
        assert result.has_operators() is False

    def test_extract_and_validate_allowed(self, parser):
        """Test extract_and_validate with allowed operators"""
        free_text, operators, warnings = parser.extract_and_validate(
            "from:alice to:bob project",
            allowed_operators=["from", "subject"]
        )
        assert free_text == "project"
        assert operators == {"from": "alice"}
        assert any("to" in w for w in warnings)

    def test_get_supported_operators(self, parser):
        """Test get_supported_operators returns all operators"""
        operators = parser.get_supported_operators()
        assert "from" in operators
        assert "to" in operators
        assert "subject" in operators
        assert "in" in operators
        assert "has" in operators
        assert "after" in operators
        assert "before" in operators
        assert "thread" in operators
        assert "cc" in operators

    def test_get_help_text(self, parser):
        """Test get_help_text returns non-empty help"""
        help_text = parser.get_help_text()
        assert len(help_text) > 100
        assert "from:" in help_text
        assert "subject:" in help_text
        assert "in:" in help_text


class TestParsedQuery:
    """Test suite for ParsedQuery dataclass"""

    def test_parsed_query_defaults(self):
        """Test ParsedQuery default values"""
        result = ParsedQuery(free_text="test")
        assert result.free_text == "test"
        assert result.operators == {}
        assert result.original_query == ""
        assert result.operator_count == 0
        assert result.parse_warnings == []

    def test_parsed_query_has_operators_with_count(self):
        """Test has_operators based on operator_count"""
        result = ParsedQuery(free_text="test", operators={"from": "alice"}, operator_count=1)
        assert result.has_operators() is True

    def test_parsed_query_get_filter_params_empty(self):
        """Test get_filter_params with no operators"""
        result = ParsedQuery(free_text="test")
        assert result.get_filter_params() == {}

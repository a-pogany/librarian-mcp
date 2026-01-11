"""
Query Operator Parser for Outlook/Gmail-style search operators.

Extracts structured operators from query strings while preserving
free-text search terms for full-text search.

Supported operators:
    - from:sender - Filter by sender (partial match)
    - to:recipient - Filter by recipient (partial match)
    - cc:recipient - Filter by CC (partial match)
    - subject:text - Filter by subject (partial match)
    - in:folder - Filter by folder/label
    - has:attachment - Filter by attachments
    - after:YYYY-MM-DD - Filter by date (after)
    - before:YYYY-MM-DD - Filter by date (before)
    - thread:id - Filter by thread ID

Example usage:
    parser = QueryOperatorParser()
    result = parser.parse("from:kiraly project deadline has:attachment")
    # result.free_text = "project deadline"
    # result.operators = {"from": "kiraly", "has": "attachment"}
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Result of parsing a query with operators."""

    # Free text for full-text search (operators removed)
    free_text: str

    # Extracted operators as {operator_name: value}
    operators: Dict[str, str] = field(default_factory=dict)

    # Original query for logging/debugging
    original_query: str = ""

    # Parsing metadata
    operator_count: int = 0
    parse_warnings: List[str] = field(default_factory=list)

    def has_operators(self) -> bool:
        """Check if any operators were extracted."""
        return self.operator_count > 0

    def get_filter_params(self) -> Dict[str, Any]:
        """
        Convert operators to search_emails filter parameters.

        Returns:
            Dictionary with filter parameter names and values
            matching the search_emails() function signature.
        """
        params: Dict[str, Any] = {}

        # Direct mappings to search_emails parameters
        if 'from' in self.operators:
            params['sender'] = self.operators['from']
        if 'to' in self.operators:
            params['recipient'] = self.operators['to']
        if 'cc' in self.operators:
            params['cc'] = self.operators['cc']
        if 'subject' in self.operators:
            params['subject_contains'] = self.operators['subject']
        if 'in' in self.operators:
            params['folder'] = self.operators['in']
        if 'thread' in self.operators:
            params['thread_id'] = self.operators['thread']
        if 'after' in self.operators:
            params['date_after'] = self.operators['after']
        if 'before' in self.operators:
            params['date_before'] = self.operators['before']

        # Boolean operators
        if 'has' in self.operators:
            value = self.operators['has'].lower()
            if value in ('attachment', 'attachments'):
                params['has_attachments'] = True

        return params


class QueryOperatorParser:
    """
    Parser for extracting search operators from query strings.

    Supports Outlook/Gmail-style operators that combine structured
    filters with free-text search in a single query string.

    Thread-safe and stateless - can be used as a singleton.
    """

    # Supported operators (lowercase)
    OPERATORS: Set[str] = {
        'from', 'to', 'cc', 'subject', 'in',
        'has', 'after', 'before', 'thread'
    }

    # Regex pattern for operator:value
    # Matches: operator:"quoted value" or operator:unquoted_value
    # Group 1: operator name
    # Group 3: quoted value (without quotes)
    # Group 4: unquoted value
    OPERATOR_PATTERN = re.compile(
        r'\b(' + '|'.join(OPERATORS) + r'):("([^"]+)"|(\S+))',
        re.IGNORECASE
    )

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse query string and extract operators.

        Operators are extracted and removed from the query,
        leaving only the free-text portion for full-text search.

        Args:
            query: Raw query string with potential operators

        Returns:
            ParsedQuery with operators extracted and free text remaining

        Examples:
            >>> parser = QueryOperatorParser()
            >>> result = parser.parse("from:kiraly project deadline")
            >>> result.free_text
            'project deadline'
            >>> result.operators
            {'from': 'kiraly'}

            >>> result = parser.parse('subject:"Q4 Report" urgent')
            >>> result.free_text
            'urgent'
            >>> result.operators
            {'subject': 'Q4 Report'}
        """
        if not query:
            return ParsedQuery(free_text="", original_query="")

        if not query.strip():
            return ParsedQuery(free_text="", original_query=query)

        operators: Dict[str, str] = {}
        warnings: List[str] = []

        # Find all operator matches
        matches = list(self.OPERATOR_PATTERN.finditer(query))

        for match in matches:
            op_name = match.group(1).lower()
            # Group 3 is quoted value (without quotes), Group 4 is unquoted
            value = match.group(3) or match.group(4)

            if not value:
                warnings.append(f"Empty value for operator '{op_name}' ignored")
                continue

            if op_name in operators:
                warnings.append(
                    f"Duplicate operator '{op_name}' - using last value"
                )

            operators[op_name] = value

        # Remove operators from query to get free text
        free_text = self.OPERATOR_PATTERN.sub('', query).strip()
        # Normalize whitespace (collapse multiple spaces)
        free_text = ' '.join(free_text.split())

        result = ParsedQuery(
            free_text=free_text,
            operators=operators,
            original_query=query,
            operator_count=len(operators),
            parse_warnings=warnings
        )

        if warnings:
            logger.debug(f"Query parse warnings: {warnings}")

        return result

    def extract_and_validate(
        self,
        query: str,
        allowed_operators: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Extract operators with optional validation against allowed list.

        Args:
            query: Raw query string
            allowed_operators: Optional whitelist of allowed operator names

        Returns:
            Tuple of (free_text, valid_operators, warnings)

        Examples:
            >>> parser = QueryOperatorParser()
            >>> text, ops, warns = parser.extract_and_validate(
            ...     "from:alice to:bob meeting",
            ...     allowed_operators=["from", "subject"]
            ... )
            >>> ops
            {'from': 'alice'}
            >>> 'to' in warns[0]
            True
        """
        parsed = self.parse(query)

        if allowed_operators:
            allowed_set = set(op.lower() for op in allowed_operators)
            invalid_ops = set(parsed.operators.keys()) - allowed_set

            for op in invalid_ops:
                parsed.parse_warnings.append(
                    f"Unsupported operator '{op}' ignored"
                )
                del parsed.operators[op]

            # Update operator count after filtering
            parsed.operator_count = len(parsed.operators)

        return (
            parsed.free_text,
            parsed.operators,
            parsed.parse_warnings
        )

    def get_supported_operators(self) -> List[str]:
        """Return list of supported operator names."""
        return sorted(self.OPERATORS)

    def get_help_text(self) -> str:
        """Return help text describing available operators."""
        return """
Supported search operators (Outlook/Gmail style):
  from:sender       Emails from sender (partial match)
  to:recipient      Emails to recipient (partial match)
  cc:recipient      Emails with CC recipient (partial match)
  subject:text      Subject contains text
  in:folder         Emails in folder (inbox, sent, important, etc.)
  has:attachment    Emails with attachments
  after:YYYY-MM-DD  Emails after date
  before:YYYY-MM-DD Emails before date
  thread:id         Emails in specific thread

Examples:
  "from:kiraly project deadline"
  "in:important has:attachment budget"
  'subject:"Q4 Report" from:finance after:2024-01-01'

Use quotes for values containing spaces: subject:"Weekly Report"
"""

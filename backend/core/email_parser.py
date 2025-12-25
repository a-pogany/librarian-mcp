"""
Email (EML) parser with preprocessing for RAG indexing

Handles:
- Quote chain removal (On ... wrote:, > prefixed, Outlook-style)
- Signature detection and removal
- Thread ID computation
- Content deduplication
- Attachment metadata extraction
"""

import email
import email.policy
import hashlib
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path

from .parsers import Parser, ParseResult

logger = logging.getLogger(__name__)


class EmailPreprocessor:
    """
    Preprocessor for email content to improve RAG quality

    Removes noise like:
    - Quoted reply chains
    - Email signatures
    - Forwarded message headers
    """

    # Quote chain patterns (multi-language support)
    QUOTE_PATTERNS = [
        # "On ... wrote:" patterns
        r'^On\s+.+\s+wrote:\s*$',
        r'^.+\s+<.+@.+>\s+wrote:\s*$',
        r'^Am\s+.+\s+schrieb\s+.+:\s*$',  # German
        r'^Le\s+.+\s+a écrit\s*:\s*$',     # French
        r'^El\s+.+\s+escribió:\s*$',       # Spanish
        r'^\d{4}\.\s*\w+\.\s*\d+\..*írta:',  # Hungarian: "2024. jan. 15. ... írta:"

        # Outlook-style headers
        r'^-+\s*Original Message\s*-+\s*$',
        r'^-+\s*Eredeti üzenet\s*-+\s*$',   # Hungarian
        r'^-+\s*Ursprüngliche Nachricht\s*-+\s*$',  # German
        r'^_{3,}\s*$',  # ___ separators

        # "From: ... Sent: ..." blocks
        r'^From:\s*.+$',
        r'^Sent:\s*.+$',
        r'^To:\s*.+$',
        r'^Cc:\s*.+$',
        r'^Subject:\s*.+$',

        # Forwarded message markers
        r'^-+\s*Forwarded message\s*-+\s*$',
        r'^Begin forwarded message:\s*$',
    ]

    # Signature markers
    SIGNATURE_MARKERS = [
        r'^--\s*$',                    # Standard -- marker
        r'^—\s*$',                     # Em-dash variant
        r'^___+\s*$',                  # Underscore line
        r'^Regards,?\s*$',
        r'^Best regards,?\s*$',
        r'^Best,?\s*$',
        r'^Thanks,?\s*$',
        r'^Thank you,?\s*$',
        r'^Sincerely,?\s*$',
        r'^Cheers,?\s*$',
        r'^Kind regards,?\s*$',
        r'^Warm regards,?\s*$',
        r'^Mit freundlichen Grüßen,?\s*$',  # German
        r'^Cordialement,?\s*$',             # French
        r'^Saludos,?\s*$',                  # Spanish
        r'^Üdvözlettel,?\s*$',              # Hungarian
        r'^Köszönettel,?\s*$',              # Hungarian
        r'^Tisztelettel,?\s*$',             # Hungarian
    ]

    # Patterns that indicate start of quote block (not just single line)
    QUOTE_BLOCK_STARTERS = [
        r'^On\s+.+\s+wrote:\s*$',
        r'^-+\s*Original Message\s*-+\s*$',
        r'^From:\s*.+\nSent:\s*.+',
    ]

    def __init__(
        self,
        remove_quotes: bool = True,
        remove_signatures: bool = True,
        min_content_length: int = 10
    ):
        """
        Initialize preprocessor

        Args:
            remove_quotes: Remove quoted reply chains
            remove_signatures: Remove email signatures
            min_content_length: Minimum content length after preprocessing
        """
        self.remove_quotes = remove_quotes
        self.remove_signatures = remove_signatures
        self.min_content_length = min_content_length

        # Compile patterns for efficiency
        self._quote_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                                for p in self.QUOTE_PATTERNS]
        self._signature_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                                    for p in self.SIGNATURE_MARKERS]
        self._quote_block_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                                      for p in self.QUOTE_BLOCK_STARTERS]

    def preprocess(self, body: str) -> Tuple[str, Dict]:
        """
        Preprocess email body

        Args:
            body: Raw email body text

        Returns:
            Tuple of (cleaned_body, preprocessing_info)
        """
        info = {
            'original_length': len(body),
            'quote_removed': False,
            'signature_removed': False,
            'quoted_lines_count': 0
        }

        if not body:
            return body, info

        cleaned = body

        # Remove quoted reply chains
        if self.remove_quotes:
            cleaned, quote_info = self._remove_quote_chains(cleaned)
            info['quote_removed'] = quote_info['removed']
            info['quoted_lines_count'] = quote_info['lines_removed']

        # Remove signature
        if self.remove_signatures:
            cleaned, sig_removed = self._remove_signature(cleaned)
            info['signature_removed'] = sig_removed

        # Clean up excessive whitespace
        cleaned = self._normalize_whitespace(cleaned)

        info['cleaned_length'] = len(cleaned)
        info['reduction_percent'] = round(
            (1 - len(cleaned) / max(len(body), 1)) * 100, 1
        )

        return cleaned, info

    def _remove_quote_chains(self, body: str) -> Tuple[str, Dict]:
        """
        Remove quoted reply chains from email body

        Strategy:
        1. Find quote block starters (On ... wrote:, Original Message, etc.)
        2. Remove everything from that point to end (quotes are usually at bottom)
        3. Also remove > prefixed lines scattered throughout
        """
        info = {'removed': False, 'lines_removed': 0}
        lines = body.split('\n')

        # Find first quote block starter
        cut_index = None
        for i, line in enumerate(lines):
            for pattern in self._quote_block_patterns:
                if pattern.match(line.strip()):
                    cut_index = i
                    break
            if cut_index is not None:
                break

        # Cut from quote block start to end
        if cut_index is not None:
            info['removed'] = True
            info['lines_removed'] = len(lines) - cut_index
            lines = lines[:cut_index]

        # Remove > prefixed lines (inline quotes)
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('>'):
                info['lines_removed'] += 1
                info['removed'] = True
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), info

    def _remove_signature(self, body: str) -> Tuple[str, bool]:
        """
        Remove email signature from body

        Strategy:
        1. Find signature marker from bottom of email
        2. Remove everything from marker to end
        """
        lines = body.split('\n')

        # Search from bottom up (signatures are at the end)
        for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
            line = lines[i].strip()

            for pattern in self._signature_patterns:
                if pattern.match(line):
                    # Found signature marker, cut here
                    return '\n'.join(lines[:i]), True

        return body, False

    def _normalize_whitespace(self, body: str) -> str:
        """Normalize excessive whitespace"""
        # Replace multiple blank lines with single blank line
        body = re.sub(r'\n{3,}', '\n\n', body)
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in body.split('\n')]
        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)


class ThreadIDGenerator:
    """Generate and manage email thread IDs"""

    @staticmethod
    def compute_thread_id(
        message_id: Optional[str],
        in_reply_to: Optional[str],
        references: Optional[str],
        subject: str
    ) -> str:
        """
        Compute thread ID for an email

        Priority:
        1. First message in References header (original thread starter)
        2. In-Reply-To header
        3. Normalized subject (fallback)

        Args:
            message_id: Message-ID header
            in_reply_to: In-Reply-To header
            references: References header
            subject: Subject line

        Returns:
            Thread ID string
        """
        # Try References first (contains full thread history)
        if references:
            # First ID in References is the thread root
            ref_ids = references.strip().split()
            if ref_ids:
                return ThreadIDGenerator._clean_message_id(ref_ids[0])

        # Try In-Reply-To
        if in_reply_to:
            return ThreadIDGenerator._clean_message_id(in_reply_to)

        # Fallback to normalized subject
        normalized = ThreadIDGenerator.normalize_subject(subject)
        return f"subject:{hashlib.md5(normalized.encode()).hexdigest()[:16]}"

    @staticmethod
    def normalize_subject(subject: str) -> str:
        """
        Normalize subject line for thread matching

        Removes:
        - Re:, Fwd:, Fw: prefixes (multi-language)
        - [tags] and (tags)
        - Extra whitespace
        """
        if not subject:
            return ""

        normalized = subject

        # Remove Re:/Fwd: prefixes (case-insensitive, multi-language)
        prefixes = [
            r'^re:\s*',
            r'^fwd?:\s*',
            r'^fw:\s*',
            r'^aw:\s*',      # German: Antwort
            r'^wg:\s*',      # German: Weitergeleitet
            r'^tr:\s*',      # French: Transféré
            r'^rép?:\s*',    # French: Réponse
            r'^rv:\s*',      # Spanish: Reenviar
            r'^vá:\s*',      # Hungarian: Válasz
            r'^továbbítva:\s*',  # Hungarian
        ]

        # Apply repeatedly (handles "Re: Re: Re: ...")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                new_normalized = re.sub(prefix, '', normalized, flags=re.IGNORECASE)
                if new_normalized != normalized:
                    normalized = new_normalized
                    changed = True

        # Remove [tags] and (tags)
        normalized = re.sub(r'\[[^\]]*\]', '', normalized)
        normalized = re.sub(r'\([^)]*\)', '', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        return normalized.strip().lower()

    @staticmethod
    def _clean_message_id(msg_id: str) -> str:
        """Clean message ID (remove < > brackets)"""
        return msg_id.strip().strip('<>').strip()


class EmailDeduplicator:
    """Handle email deduplication across multiple PST/mailbox sources"""

    def __init__(self):
        self._seen_hashes: Set[str] = set()

    def compute_hash(
        self,
        message_id: Optional[str],
        body: str,
        subject: str,
        date: Optional[str]
    ) -> str:
        """
        Compute content hash for deduplication

        Uses combination of:
        - Message-ID (if available)
        - Body content
        - Subject
        - Date

        Args:
            message_id: Message-ID header
            body: Email body (after preprocessing)
            subject: Subject line
            date: Date string

        Returns:
            SHA-256 hash string
        """
        # If we have Message-ID, use it as primary identifier
        if message_id:
            content = f"msgid:{message_id}"
        else:
            # Fallback to content-based hash
            normalized_body = ' '.join(body.lower().split())[:1000]  # First 1000 chars
            normalized_subject = ThreadIDGenerator.normalize_subject(subject)
            content = f"content:{normalized_subject}:{date or ''}:{normalized_body}"

        return hashlib.sha256(content.encode()).hexdigest()

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if email with this hash was already seen"""
        return content_hash in self._seen_hashes

    def mark_seen(self, content_hash: str):
        """Mark hash as seen"""
        self._seen_hashes.add(content_hash)

    def clear(self):
        """Clear seen hashes (for new indexing session)"""
        self._seen_hashes.clear()

    def get_stats(self) -> Dict:
        """Get deduplication stats"""
        return {'unique_emails_seen': len(self._seen_hashes)}


class EMLParser(Parser):
    """
    Parse EML files with email-specific preprocessing

    Features:
    - Quote chain removal
    - Signature detection
    - Thread ID extraction
    - Deduplication support
    - Attachment metadata
    """

    # Shared deduplicator across all parser instances
    _deduplicator = EmailDeduplicator()

    def __init__(
        self,
        remove_quotes: bool = True,
        remove_signatures: bool = True,
        extract_attachments: bool = True,
        skip_duplicates: bool = True
    ):
        """
        Initialize EML parser

        Args:
            remove_quotes: Remove quoted reply chains
            remove_signatures: Remove email signatures
            extract_attachments: Extract attachment metadata
            skip_duplicates: Skip duplicate emails (same content hash)
        """
        self.preprocessor = EmailPreprocessor(
            remove_quotes=remove_quotes,
            remove_signatures=remove_signatures
        )
        self.extract_attachments = extract_attachments
        self.skip_duplicates = skip_duplicates

    def parse(self, file_path: str) -> ParseResult:
        """
        Parse EML file

        Args:
            file_path: Path to .eml file

        Returns:
            ParseResult with cleaned content and email metadata
        """
        # Read and parse email
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)

        # Extract headers
        headers = self._extract_headers(msg)

        # Extract body
        raw_body = self._extract_body(msg)

        # Preprocess body
        cleaned_body, preprocess_info = self.preprocessor.preprocess(raw_body)

        # Compute thread ID
        thread_id = ThreadIDGenerator.compute_thread_id(
            message_id=headers.get('message_id'),
            in_reply_to=headers.get('in_reply_to'),
            references=headers.get('references'),
            subject=headers.get('subject', '')
        )

        # Compute content hash for deduplication
        content_hash = self._deduplicator.compute_hash(
            message_id=headers.get('message_id'),
            body=cleaned_body,
            subject=headers.get('subject', ''),
            date=headers.get('date')
        )

        # Check for duplicates
        is_duplicate = self._deduplicator.is_duplicate(content_hash)
        if not is_duplicate:
            self._deduplicator.mark_seen(content_hash)

        # Extract attachments
        attachments = []
        if self.extract_attachments:
            attachments = self._extract_attachment_metadata(msg)

        # Add lightweight search hints so filenames/subjects are searchable
        cleaned_body = self._append_search_hints(
            cleaned_body,
            headers=headers,
            attachments=attachments
        )

        # Normalize subject
        normalized_subject = ThreadIDGenerator.normalize_subject(headers.get('subject', ''))

        # Build metadata
        metadata = {
            'doc_type': 'email',
            'message_id': headers.get('message_id'),
            'thread_id': thread_id,
            'from': headers.get('from'),
            'to': headers.get('to'),
            'cc': headers.get('cc'),
            'date': headers.get('date'),
            'subject': headers.get('subject'),
            'subject_normalized': normalized_subject,
            'in_reply_to': headers.get('in_reply_to'),
            'has_attachments': len(attachments) > 0,
            'attachment_count': len(attachments),
            'attachments': attachments,
            'content_hash': content_hash,
            'is_duplicate': is_duplicate,
            'preprocessing': preprocess_info,
            'encoding': 'utf-8'
        }

        # Use subject as heading
        headings = [headers.get('subject', 'No Subject')]

        # Return empty content for duplicates if skip is enabled
        if is_duplicate and self.skip_duplicates:
            logger.debug(f"Skipping duplicate email: {headers.get('subject', 'No Subject')}")
            return ParseResult(
                content='',  # Empty content signals skip
                headings=headings,
                metadata=metadata
            )

        return ParseResult(
            content=cleaned_body,
            headings=headings,
            metadata=metadata
        )

    def _append_search_hints(
        self,
        content: str,
        headers: Dict,
        attachments: List[Dict]
    ) -> str:
        """Append subject and attachment filenames to content for indexing."""
        hints = []

        subject = headers.get('subject')
        if subject:
            hints.append(f"Subject: {subject}")

        from_header = headers.get('from')
        if from_header:
            hints.append(f"From: {from_header}")

        to_header = headers.get('to')
        if to_header:
            if isinstance(to_header, list):
                hints.append(f"To: {', '.join(to_header)}")
            else:
                hints.append(f"To: {to_header}")

        attachment_names = [a.get('filename') for a in attachments if a.get('filename')]
        if attachment_names:
            hints.append(f"Attachments: {', '.join(attachment_names)}")

        if not hints:
            return content

        separator = "\n\n" if content else ""
        return f"{content}{separator}{'\n'.join(hints)}"

    def _extract_headers(self, msg: EmailMessage) -> Dict:
        """Extract email headers"""
        headers = {}

        # Standard headers
        headers['message_id'] = self._safe_header_string(msg, 'Message-ID').strip()
        headers['subject'] = self._safe_header_string(msg, 'Subject')
        headers['from'] = self._safe_header_string(msg, 'From')
        headers['date'] = self._safe_header_string(msg, 'Date')
        headers['in_reply_to'] = self._safe_header_string(msg, 'In-Reply-To').strip()
        headers['references'] = self._safe_header_string(msg, 'References').strip()

        # To/CC can have multiple recipients
        headers['to'] = self._parse_address_list(self._safe_header_string(msg, 'To'))
        headers['cc'] = self._parse_address_list(self._safe_header_string(msg, 'Cc'))

        # Parse date to ISO format
        if headers['date']:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(headers['date'])
                headers['date_iso'] = dt.isoformat()
            except Exception:
                headers['date_iso'] = None

        return headers

    def _safe_header_string(self, msg: EmailMessage, name: str) -> str:
        """Safely read and stringify a header value."""
        try:
            value = msg.get(name, '')
        except Exception as e:
            logger.debug(f"Failed to read header {name}: {e}")
            return ''

        if value is None:
            return ''
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception as e:
            logger.debug(f"Failed to stringify header {name}: {e}")
            return ''

    def _parse_address_list(self, header: str) -> List[str]:
        """Parse comma-separated address list"""
        if not header:
            return []

        try:
            from email.headerregistry import Group

            if hasattr(header, 'addresses'):
                addresses = []
                for entry in header.addresses:
                    if isinstance(entry, Group) or hasattr(entry, 'addresses'):
                        for addr in entry.addresses:
                            formatted = self._format_address(addr)
                            if formatted:
                                addresses.append(formatted)
                    else:
                        formatted = self._format_address(entry)
                        if formatted:
                            addresses.append(formatted)
                return addresses
        except Exception as e:
            logger.debug(f"Header registry parse failed: {e}")

        try:
            header_str = str(header)
        except Exception:
            return []

        addresses = [addr.strip() for addr in header_str.split(',')]
        return [addr for addr in addresses if addr]

    def _format_address(self, addr) -> str:
        """Format headerregistry Address into display string."""
        if addr is None:
            return ""

        try:
            from email.headerregistry import Group
            if isinstance(addr, Group) or hasattr(addr, 'addresses'):
                group_addresses = []
                for entry in getattr(addr, 'addresses', []) or []:
                    formatted = self._format_address(entry)
                    if formatted:
                        group_addresses.append(formatted)
                display_name = getattr(addr, 'display_name', '') or ''
                if group_addresses:
                    if display_name:
                        return f"{display_name}: {', '.join(group_addresses)}"
                    return ', '.join(group_addresses)
                return display_name
        except Exception:
            pass

        try:
            display_name = getattr(addr, 'display_name', '') or ''
            addr_spec = getattr(addr, 'addr_spec', '') or ''

            if not addr_spec:
                username = getattr(addr, 'username', '') or ''
                domain = getattr(addr, 'domain', '') or ''
                if username and domain:
                    addr_spec = f"{username}@{domain}"
                elif username:
                    addr_spec = username
                else:
                    local_part = getattr(addr, 'local_part', '') or ''
                    domain = getattr(addr, 'domain', '') or ''
                    if local_part and domain:
                        addr_spec = f"{local_part}@{domain}"
                    elif local_part:
                        addr_spec = local_part

            if display_name and addr_spec:
                return f"{display_name} <{addr_spec}>"
            if addr_spec:
                return addr_spec
            return display_name
        except Exception:
            try:
                return str(addr)
            except Exception:
                return ""

    def _extract_body(self, msg: EmailMessage) -> str:
        """
        Extract email body text

        Handles:
        - Plain text emails
        - HTML-only emails (strips HTML)
        - Multipart emails (prefers plain text)
        """
        body = ""

        if msg.is_multipart():
            # Walk through parts, prefer plain text
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))

                # Skip attachments
                if 'attachment' in content_disposition:
                    continue

                if content_type == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body = payload.decode(charset, errors='replace')
                            break  # Found plain text, use it
                        except Exception:
                            pass

                elif content_type == 'text/html' and not body:
                    # Fallback to HTML if no plain text
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            html = payload.decode(charset, errors='replace')
                            body = self._html_to_text(html)
                        except Exception:
                            pass
        else:
            # Non-multipart
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)

            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    text = payload.decode(charset, errors='replace')
                    if content_type == 'text/html':
                        body = self._html_to_text(text)
                    else:
                        body = text
                except Exception:
                    pass

        return body

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML to plain text

        Simple implementation - strips tags and normalizes whitespace
        """
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Replace common HTML elements with text equivalents
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<p[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</p>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<div[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</div>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<li[^>]*>', '\n• ', html, flags=re.IGNORECASE)

        # Strip remaining tags
        html = re.sub(r'<[^>]+>', '', html)

        # Decode HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')

        # Normalize whitespace
        lines = [line.strip() for line in html.split('\n')]
        text = '\n'.join(line for line in lines if line)

        return text

    def _extract_attachment_metadata(self, msg: EmailMessage) -> List[Dict]:
        """
        Extract metadata about attachments (not content)

        Returns list of attachment info dicts
        """
        attachments = []

        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            content_disposition = str(part.get('Content-Disposition', ''))

            if 'attachment' in content_disposition or part.get_filename():
                filename = part.get_filename() or 'unnamed'
                content_type = part.get_content_type()

                # Get size if possible
                payload = part.get_payload(decode=True)
                size = len(payload) if payload else 0

                attachments.append({
                    'filename': filename,
                    'content_type': content_type,
                    'size_bytes': size,
                    'size_human': self._format_size(size)
                })

        return attachments

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable form"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    @classmethod
    def reset_deduplicator(cls):
        """Reset the deduplicator (call before new indexing run)"""
        cls._deduplicator.clear()

    @classmethod
    def get_dedup_stats(cls) -> Dict:
        """Get deduplication statistics"""
        return cls._deduplicator.get_stats()

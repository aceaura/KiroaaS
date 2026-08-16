# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Parsers for AWS Event Stream format.

Contains classes and functions for:
- Parsing binary AWS SSE stream
- Extracting JSON events
- Processing tool calls
- Content deduplication
"""

import json
import re
import struct
import zlib
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.utils import generate_tool_call_id


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Finds the position of the closing brace considering nesting and strings.
    
    Uses bracket counting for correct parsing of nested JSON.
    Accounts for quoted strings and escape sequences.
    
    Args:
        text: Text to search
        start_pos: Position of opening brace '{'
    
    Returns:
        Position of closing brace or -1 if not found
    
    Example:
        >>> find_matching_brace('{"a": {"b": 1}}', 0)
        14
        >>> find_matching_brace('{"a": "{}"}', 0)
        10
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
    
    return -1


def parse_bracket_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parses tool calls in [Called func_name with args: {...}] format.
    
    Some models return tool calls in text format instead of
    structured JSON. This function extracts them.
    
    Args:
        response_text: Model response text
    
    Returns:
        List of tool calls in OpenAI format
    
    Example:
        >>> text = "[Called get_weather with args: {\"city\": \"London\"}]"
        >>> calls = parse_bracket_tool_calls(text)
        >>> calls[0]["function"]["name"]
        'get_weather'
    """
    if not response_text or "[Called" not in response_text:
        return []
    
    tool_calls = []
    pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*'
    
    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        func_name = match.group(1)
        args_start = match.end()
        
        # Find JSON start
        json_start = response_text.find('{', args_start)
        if json_start == -1:
            continue
        
        # Find JSON end considering nesting
        json_end = find_matching_brace(response_text, json_start)
        if json_end == -1:
            continue
        
        json_str = response_text[json_start:json_end + 1]
        
        try:
            args = json.loads(json_str)
            tool_call_id = generate_tool_call_id()
            # index will be added later when forming the final response
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {json_str[:100]}")
    
    return tool_calls


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicate tool calls.
    
    Deduplication occurs by two criteria:
    1. By id - if there are multiple tool calls with the same id, keep the one with
       more arguments (not empty "{}")
    2. By name+arguments - remove complete duplicates
    
    Args:
        tool_calls: List of tool calls
    
    Returns:
        List of unique tool calls
    """
    # First deduplicate by id - keep tool call with non-empty arguments
    by_id: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not tc_id:
            # Without id - add as is (will be deduplicated by name+args)
            continue
        
        existing = by_id.get(tc_id)
        if existing is None:
            by_id[tc_id] = tc
        else:
            # Duplicate by id exists - keep the one with more arguments
            existing_args = existing.get("function", {}).get("arguments", "{}")
            current_args = tc.get("function", {}).get("arguments", "{}")
            
            # Prefer non-empty arguments
            if current_args != "{}" and (existing_args == "{}" or len(current_args) > len(existing_args)):
                logger.debug(f"Replacing tool call {tc_id} with better arguments: {len(existing_args)} -> {len(current_args)}")
                by_id[tc_id] = tc
    
    # Collect tool calls: first those with id, then without id
    result_with_id = list(by_id.values())
    result_without_id = [tc for tc in tool_calls if not tc.get("id")]
    
    # Now deduplicate by name+arguments for all
    seen = set()
    unique = []
    
    for tc in result_with_id + result_without_id:
        # Protection against None in function
        func = tc.get("function") or {}
        func_name = func.get("name") or ""
        func_args = func.get("arguments") or "{}"
        key = f"{func_name}-{func_args}"
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    
    if len(tool_calls) != len(unique):
        logger.debug(f"Deduplicated tool calls: {len(tool_calls)} -> {len(unique)}")
    
    return unique


class AwsEventStreamParser:
    """
    Parser for AWS Event Stream format.
    
    AWS returns events in binary format with :message-type...event delimiters.
    This class extracts JSON events from the stream and converts them to a convenient format.
    
    Supported event types:
    - content: Text content of response
    - tool_start: Start of tool call (name, toolUseId)
    - tool_input: Continuation of input for tool call
    - tool_stop: End of tool call
    - usage: Credit consumption information
    - context_usage: Context usage percentage
    
    Attributes:
        buffer: Buffer for accumulating data
        last_content: Last processed content (for deduplication)
        current_tool_call: Current incomplete tool call
        tool_calls: List of completed tool calls
    
    Example:
        >>> parser = AwsEventStreamParser()
        >>> events = parser.feed(chunk)
        >>> for event in events:
        ...     if event["type"] == "content":
        ...         print(event["data"])
    """
    
    # Patterns for finding JSON events
    EVENT_PATTERNS = [
        ('{"content":', 'content'),
        ('{"name":', 'tool_start'),
        ('{"input":', 'tool_input'),
        ('{"stop":', 'tool_stop'),
        ('{"followupPrompt":', 'followup'),
        ('{"usage":', 'usage'),
        ('{"contextUsagePercentage":', 'context_usage'),
    ]
    
    _MAX_FRAME_SIZE = 4 * 1024 * 1024
    _HEADER_VALUE_SIZES = {
        0: 0,  # bool true
        1: 0,  # bool false
        2: 1,  # byte
        3: 2,  # short
        4: 4,  # int
        5: 8,  # long
        6: -1,  # byte array
        7: -1,  # string
        8: 8,  # timestamp
        9: 16,  # uuid
    }

    def __init__(self):
        """Initializes the parser."""
        self.buffer = ""
        self._binary_buffer = bytearray()
        self._stream_mode: Optional[str] = None
        self.last_content: Optional[str] = None  # For deduplicating repeating content
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Adds chunk to buffer and returns parsed events.

        AWS Event Stream frames are decoded using their prelude and ``:event-type``
        header. Plain JSON scanning is retained as a compatibility fallback.

        Args:
            chunk: Bytes of data from stream

        Returns:
            List of events in {"type": str, "data": Any} format
        """
        if self._stream_mode != "text":
            self._binary_buffer.extend(chunk)
            if self._stream_mode is None:
                mode = self._detect_stream_mode()
                if mode is None:
                    return []
                self._stream_mode = mode

            if self._stream_mode == "aws":
                return self._feed_aws_frames()

            chunk = bytes(self._binary_buffer)
            self._binary_buffer.clear()

        return self._feed_text(chunk)

    def _detect_stream_mode(self) -> Optional[str]:
        """Detect AWS framing without misclassifying incomplete binary data."""
        if not self._binary_buffer:
            return None
        if self._binary_buffer[0] != 0:
            return "text"
        if len(self._binary_buffer) < 12:
            return None

        total_length, headers_length, prelude_crc = struct.unpack(
            ">III", self._binary_buffer[:12]
        )
        if not (
            16 <= total_length <= self._MAX_FRAME_SIZE
            and headers_length <= total_length - 16
            and zlib.crc32(self._binary_buffer[:8]) & 0xFFFFFFFF == prelude_crc
        ):
            return "text"
        return "aws"

    def _feed_aws_frames(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while len(self._binary_buffer) >= 12:
            total_length, headers_length, prelude_crc = struct.unpack(
                ">III", self._binary_buffer[:12]
            )
            if not (
                16 <= total_length <= self._MAX_FRAME_SIZE
                and headers_length <= total_length - 16
                and zlib.crc32(self._binary_buffer[:8]) & 0xFFFFFFFF == prelude_crc
            ):
                logger.warning("Invalid AWS Event Stream prelude; dropping buffered frame")
                self._binary_buffer.clear()
                break
            if len(self._binary_buffer) < total_length:
                break

            frame = bytes(self._binary_buffer[:total_length])
            del self._binary_buffer[:total_length]
            expected_crc = struct.unpack(">I", frame[-4:])[0]
            if zlib.crc32(frame[:-4]) & 0xFFFFFFFF != expected_crc:
                logger.warning("Invalid AWS Event Stream message CRC; dropping frame")
                continue

            headers = frame[12:12 + headers_length]
            payload = frame[12 + headers_length:-4]
            event_type = self._extract_event_type(headers)
            event = self._process_aws_event(event_type, payload)
            if event:
                events.append(event)
        return events

    def _extract_event_type(self, headers: bytes) -> Optional[str]:
        """Extract the string ``:event-type`` header from an AWS frame."""
        offset = 0
        while offset < len(headers):
            name_length = headers[offset]
            offset += 1
            if offset + name_length + 1 > len(headers):
                return None
            name = headers[offset:offset + name_length].decode("utf-8", errors="ignore")
            offset += name_length
            value_type = headers[offset]
            offset += 1
            value_size = self._HEADER_VALUE_SIZES.get(value_type)
            if value_size is None:
                return None
            if value_size < 0:
                if offset + 2 > len(headers):
                    return None
                value_size = struct.unpack(">H", headers[offset:offset + 2])[0]
                offset += 2
            if offset + value_size > len(headers):
                return None
            value = headers[offset:offset + value_size]
            offset += value_size
            if name == ":event-type" and value_type == 7:
                return value.decode("utf-8", errors="ignore")
        return None

    def _process_aws_event(
        self,
        event_type: Optional[str],
        payload: bytes,
    ) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(f"Failed to parse AWS event payload type={event_type or 'unknown'}")
            return None
        if not isinstance(data, dict):
            return None

        if event_type == "assistantResponseEvent":
            return self._process_content_event(data)
        if event_type == "meteringEvent":
            return self._process_event(data, "usage")
        if event_type == "contextUsageEvent":
            return self._process_event(data, "context_usage")
        if event_type == "toolUseEvent":
            if "name" in data:
                return self._process_event(data, "tool_start")
            if "input" in data:
                return self._process_event(data, "tool_input")
            if "stop" in data:
                return self._process_event(data, "tool_stop")
        return None

    def _feed_text(self, chunk: bytes) -> List[Dict[str, Any]]:
        try:
            self.buffer += chunk.decode('utf-8', errors='ignore')
        except Exception:
            return []

        events = []

        while True:
            # Find nearest pattern
            earliest_pos = -1
            earliest_type = None

            for pattern, event_type in self.EVENT_PATTERNS:
                pos = self.buffer.find(pattern)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos
                    earliest_type = event_type

            if earliest_pos == -1:
                break

            # Find JSON end
            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                # JSON not complete, wait for more data
                break

            json_str = self.buffer[earliest_pos:json_end + 1]
            self.buffer = self.buffer[json_end + 1:]

            try:
                data = json.loads(json_str)
                event = self._process_event(data, earliest_type)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {json_str[:100]}")

        return events
    
    def _process_event(self, data: dict, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Processes a parsed event.
        
        Args:
            data: Parsed JSON
            event_type: Event type
        
        Returns:
            Processed event or None
        """
        if event_type == 'content':
            return self._process_content_event(data)
        elif event_type == 'tool_start':
            return self._process_tool_start_event(data)
        elif event_type == 'tool_input':
            return self._process_tool_input_event(data)
        elif event_type == 'tool_stop':
            return self._process_tool_stop_event(data)
        elif event_type == 'usage':
            return {"type": "usage", "data": data.get('usage', 0)}
        elif event_type == 'context_usage':
            return {"type": "context_usage", "data": data.get('contextUsagePercentage', 0)}
        
        return None
    
    def _process_content_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes content event."""
        content = data.get('content', '')
        
        # Skip followupPrompt
        if data.get('followupPrompt'):
            return None
        
        # Deduplicate repeating content
        if content == self.last_content:
            return None
        
        self.last_content = content
        
        return {"type": "content", "data": content}
    
    def _process_tool_start_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call start."""
        # Finalize previous tool call if exists
        if self.current_tool_call:
            self._finalize_tool_call()
        
        # input can be string or object
        input_data = data.get('input', '')
        if isinstance(input_data, dict):
            if input_data:
                # Non-empty dict: serialize it
                input_str = json.dumps(input_data)
            else:
                # Empty dict {}: fragments will follow, use empty string
                input_str = ''
        else:
            input_str = str(input_data) if input_data else ''
        
        self.current_tool_call = {
            "id": data.get('toolUseId', generate_tool_call_id()),
            "type": "function",
            "function": {
                "name": data.get('name', ''),
                "arguments": input_str
            }
        }
        
        if data.get('stop'):
            self._finalize_tool_call()
        
        return None
    
    def _process_tool_input_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes input continuation for tool call."""
        if self.current_tool_call:
            # input can be string or object
            input_data = data.get('input', '')
            if isinstance(input_data, dict):
                if input_data:
                    input_str = json.dumps(input_data)
                else:
                    input_str = ''
            else:
                input_str = str(input_data) if input_data else ''
            self.current_tool_call['function']['arguments'] += input_str
        return None
    
    def _process_tool_stop_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call end."""
        if self.current_tool_call and data.get('stop'):
            self._finalize_tool_call()
        return None
    
    def _finalize_tool_call(self) -> None:
        """Finalizes current tool call and adds to list."""
        if not self.current_tool_call:
            return
        
        # Try to parse and normalize arguments as JSON
        args = self.current_tool_call['function']['arguments']
        tool_name = self.current_tool_call['function'].get('name', 'unknown')
        
        logger.debug(f"Finalizing tool call '{tool_name}' with raw arguments: {repr(args)[:200]}")
        
        if isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    # Ensure result is a JSON string
                    self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                    logger.debug(f"Tool '{tool_name}' arguments parsed successfully: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                except json.JSONDecodeError as e:
                    # Analyze the failure to provide better diagnostics
                    truncation_info = self._diagnose_json_truncation(args)
                    
                    if truncation_info["is_truncated"]:
                        # Mark for recovery system
                        self.current_tool_call['_truncation_detected'] = True
                        self.current_tool_call['_truncation_info'] = truncation_info
                        
                        # Check if recovery is enabled
                        from kiro.config import TRUNCATION_RECOVERY
                        tool_id = self.current_tool_call.get('id', 'unknown')
                        
                        # Clear error message: this is Kiro API's fault, not ours
                        logger.error(
                            f"Tool call truncated by Kiro API: "
                            f"tool='{tool_name}', id={tool_id}, size={truncation_info['size_bytes']} bytes, "
                            f"reason={truncation_info['reason']}. "
                            f"This is a Kiro API limitation. "
                            f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
                        )
                    else:
                        # Regular JSON parse error
                        logger.warning(f"Failed to parse tool '{tool_name}' arguments: {e}. Raw: {args[:200]}")
                    
                    self.current_tool_call['function']['arguments'] = "{}"
            else:
                # Empty string - use empty object
                # This is normal behavior for duplicate tool calls from Kiro
                logger.debug(f"Tool '{tool_name}' has empty arguments string (will be deduplicated)")
                self.current_tool_call['function']['arguments'] = "{}"
        elif isinstance(args, dict):
            # If already an object - serialize to string
            self.current_tool_call['function']['arguments'] = json.dumps(args)
            logger.debug(f"Tool '{tool_name}' arguments already dict with keys: {list(args.keys())}")
        else:
            # Unknown type - empty object
            logger.warning(f"Tool '{tool_name}' has unexpected arguments type: {type(args)}")
            self.current_tool_call['function']['arguments'] = "{}"
        
        self.tool_calls.append(self.current_tool_call)
        self.current_tool_call = None
    
    def _diagnose_json_truncation(self, json_str: str) -> Dict[str, Any]:
        """
        Analyzes a malformed JSON string to determine if it was truncated.
        
        This helps distinguish between upstream issues (Kiro API cutting off
        large tool call arguments) and actual malformed JSON from the model.
        
        Args:
            json_str: The raw JSON string that failed to parse
        
        Returns:
            Dictionary with diagnostic information:
            - is_truncated: True if the JSON appears to be cut off
            - reason: Human-readable explanation of why it's truncated
            - size_bytes: Size of the received data
        """
        size_bytes = len(json_str.encode('utf-8'))
        stripped = json_str.strip()
        
        # Check for obvious truncation signs
        if not stripped:
            return {"is_truncated": False, "reason": "empty string", "size_bytes": size_bytes}
        
        # Count braces and brackets (simplified, doesn't account for strings perfectly)
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        open_brackets = stripped.count('[')
        close_brackets = stripped.count(']')
        
        # Check if JSON starts with { but doesn't end with }
        if stripped.startswith('{') and not stripped.endswith('}'):
            missing = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing brace(s)",
                "size_bytes": size_bytes
            }
        
        # Check if JSON starts with [ but doesn't end with ]
        if stripped.startswith('[') and not stripped.endswith(']'):
            missing = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing bracket(s)",
                "size_bytes": size_bytes
            }
        
        # Check for unbalanced braces/brackets
        if open_braces != close_braces:
            diff = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"unbalanced braces ({open_braces} open, {close_braces} close)",
                "size_bytes": size_bytes
            }
        
        if open_brackets != close_brackets:
            diff = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"unbalanced brackets ({open_brackets} open, {close_brackets} close)",
                "size_bytes": size_bytes
            }
        
        # Check for unclosed string (ends with backslash or inside quotes)
        # This is a heuristic - count unescaped quotes
        quote_count = 0
        i = 0
        while i < len(stripped):
            if stripped[i] == '\\' and i + 1 < len(stripped):
                i += 2  # Skip escaped character
                continue
            if stripped[i] == '"':
                quote_count += 1
            i += 1
        
        if quote_count % 2 != 0:
            return {
                "is_truncated": True,
                "reason": "unclosed string literal",
                "size_bytes": size_bytes
            }
        
        # Doesn't look truncated, probably just malformed
        return {"is_truncated": False, "reason": "malformed JSON", "size_bytes": size_bytes}
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Returns all collected tool calls.
        
        Finalizes current tool call if not finished.
        Removes duplicates.
        
        Returns:
            List of unique tool calls
        """
        if self.current_tool_call:
            self._finalize_tool_call()
        return deduplicate_tool_calls(self.tool_calls)
    
    def reset(self) -> None:
        """Resets parser state."""
        self.buffer = ""
        self._binary_buffer.clear()
        self._stream_mode = None
        self.last_content = None
        self.current_tool_call = None
        self.tool_calls = []

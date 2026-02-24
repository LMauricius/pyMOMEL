"""
MOMEL parser — Mauricio's Obvious Minimal Expandable Language.

Public API:
    parse(text: str) -> dict
    parse_file(path) -> dict
    Number(value, unit)
    ParseError
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Number:
    """A MOMEL number value with an optional unit suffix."""

    value: int | float
    # Empty string when no suffix was present.
    unit: str = ""


class ParseError(Exception):
    """Syntax error raised with source location context."""

    def __init__(self, msg: str, line: int, col: int) -> None:
        super().__init__(f"line {line}, col {col}: {msg}")
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Character-class / escape utilities
# ---------------------------------------------------------------------------

# Single-character escape letter -> decoded character.
_ESCAPE_MAP: dict[str, str] = {
    "\n": "",
    "n": "\n",
    "_": " ",
    ":": ":",
    "'": "'",
    '"': '"',
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "{": "{",
    "}": "}",
    "t": "\t",
    "v": "\v",
    "r": "\r",
    "b": "\b",
    "a": "\a",
    "f": "\f",
    "e": "\x1b",
    "0": "\x00",
    "\\": "\\",
}

# Characters that start/delimit special MOMEL values.
_STRUCTURAL = set("()[]{}\"' \t\n")

_WHITESPACE = set(" \t")

_SIGN_CHARS = set("+-")
_HEX_DIGITS = set("0123456789abcdefABCDEF")
_DEC_DIGITS = set("0123456789")
_OCT_DIGITS = set("01234567")
_BIN_DIGITS = set("01")
_HEX_MARK = set("xX")
_OCT_MARK = set("oO")
_BIN_MARK = set("bB")


def decode_escape(src: str, pos: int, line: int, col: int) -> tuple[str, int]:
    """
    Parse one escape sequence at src[pos] (the char *after* the backslash).
    Returns (decoded_char, new_pos).
    Raises ParseError on unknown or truncated sequences.
    """
    if pos >= len(src):
        raise ParseError("truncated escape at EOF", line, col)

    ch = src[pos]

    if ch in _ESCAPE_MAP:
        return _ESCAPE_MAP[ch], pos + 1

    # \xHH
    if ch == "x":
        hex_str = src[pos + 1 : pos + 3]
        if len(hex_str) < 2 or not all(c in _HEX_DIGITS for c in hex_str):
            raise ParseError(r"expected 2 hex digits after \x", line, col)
        return chr(int(hex_str, 16)), pos + 3

    # \uHHHH
    if ch == "u":
        hex_str = src[pos + 1 : pos + 5]
        if len(hex_str) < 4 or not all(c in _HEX_DIGITS for c in hex_str):
            raise ParseError(r"expected 4 hex digits after \u", line, col)
        return chr(int(hex_str, 16)), pos + 5

    # \UHHHHHHHH
    if ch == "U":
        hex_str = src[pos + 1 : pos + 9]
        if len(hex_str) < 8 or not all(c in _HEX_DIGITS for c in hex_str):
            raise ParseError(r"expected 8 hex digits after \U", line, col)
        codepoint = int(hex_str, 16)
        if codepoint > 0x10FFFF:
            raise ParseError(
                f"Unicode codepoint {codepoint:#x} out of range", line, col
            )
        return chr(codepoint), pos + 9

    raise ParseError(f"unknown escape sequence: \\{ch}", line, col)


def is_suffix_char(ch: str) -> bool:
    """True if ch may appear unescaped in a number suffix or simple string."""
    return ch not in _STRUCTURAL and ch != ""


def is_identifier_char(ch: str) -> bool:
    """True if ch may appear unescaped in a dict key (not ':' and not newline)."""
    return ch != ":" and ch != "\n" and ch != ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    """Stateful recursive-descent MOMEL parser."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0
        self.line = 1
        # Byte offset where the current line started (for column tracking).
        self.line_start = 0

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def peek(self, offset: int = 0) -> str:
        """Return char at pos+offset without consuming, or '' at EOF."""
        idx = self.pos + offset
        return self.src[idx] if idx < len(self.src) else ""

    def advance(self) -> str:
        """Consume and return the current char; update line tracking."""
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.line_start = self.pos
        return ch

    def advance_escape(self) -> str:
        """Consume an escape sequence; update line tracking."""
        decoded, new_pos = decode_escape(
            self.src, self.pos, self.line, self.current_col()
        )

        for cur_pos in range(self.pos, new_pos):
            ch = self.src[cur_pos]
            if ch == "\n":
                self.line += 1
                self.line_start = cur_pos + 1

        self.pos = new_pos
        return decoded

    def current_col(self) -> int:
        """0-based column of the current position."""
        return self.pos - self.line_start

    def error(self, msg: str) -> ParseError:
        return ParseError(msg, self.line, self.current_col() + 1)

    def expect(self, ch: str) -> None:
        """Consume ch or raise ParseError."""
        if self.peek() != ch:
            got = repr(self.peek()) if self.peek() else "EOF"
            raise self.error(f"expected {ch!r}, got {got}")
        self.advance()

    def skip_ws_inline(self) -> None:
        """Skip spaces and tabs on the current line only."""
        while self.peek() in _WHITESPACE or (
            self.peek() == "\\" and self.peek(1) == "\n"
        ):
            if self.peek() == "\\":
                self.advance()
            self.advance()

    def skip_comment(self) -> None:
        """If at '#', advance to (not past) the newline/EOF."""
        if self.peek() == "#":
            while self.peek() not in ("\n", ""):
                self.advance()

    def skip_blank_lines(self) -> None:
        """Skip lines that are entirely whitespace or comments."""
        while True:
            self.skip_ws_inline()
            self.skip_comment()

            if self.peek() == "\n":
                self.advance()  # consume the newline and keep going
            elif self.peek() == "":
                break  # EOF is fine
            else:
                break

    # ------------------------------------------------------------------
    # Top-level / dict
    # ------------------------------------------------------------------

    def parse_top_level(self) -> dict:
        """Parse the entire file as a dict body (no enclosing braces)."""
        result = self._parse_dict_body(closing=None)
        if self.peek() != "":
            raise self.error(f"unexpected character {self.peek()!r}")
        return result

    def _parse_dict_body(self, closing: str | None) -> dict:
        """
        Parse key:value pairs until closing char ('}') or EOF (if closing=None).
        Handles duplicate dict-only keys by merging.
        """
        result: dict = {}

        while True:
            self.skip_blank_lines()

            ch = self.peek()

            # End of dict body.
            if closing is not None:
                if ch == "":
                    raise self.error(f"unexpected EOF, expected {closing!r}")
                if ch == closing:
                    self.advance()
                    return result
            elif ch == "":
                return result

            # Parse one field.
            key = self._parse_identifier()
            self.expect(":")
            self.skip_ws_inline()
            val = self._parse_tuple()

            # Duplicate key handling: merge if both sides are (dict,).
            if key in result:
                old_val = result[key]
                if (
                    len(old_val) == 1
                    and isinstance(old_val[0], dict)
                    and len(val) == 1
                    and isinstance(val[0], dict)
                ):
                    result[key] = (_merge_dicts(old_val[0], val[0]),)
                else:
                    raise self.error(f"duplicate key {key!r}")
            else:
                result[key] = val

            self.skip_comment()

            # Expect end of line, EOF, or closing brace.
            if self.peek() != "\n" and self.peek() != "":
                raise self.error(f"Expected end of line, got {self.peek()!r}")

    def _parse_identifier(self) -> str:
        """
        Consume chars until unescaped ':' or newline.
        Apply escape sequences; strip leading/trailing ASCII whitespace.
        \\_ escape -> literal space (preserved, not stripped).
        """
        parts: list[str] = []
        numTrailingWS = 0

        while True:
            ch = self.peek()

            if ch == "\\":
                self.advance()  # consume '\'
                parts.append(self.advance_escape())
                numTrailingWS = 0
            elif is_identifier_char(ch):
                parts.append(ch)
                self.advance()

                if ch in _WHITESPACE:
                    numTrailingWS += 1
                else:
                    numTrailingWS = 0
            else:
                break

        raw = "".join(parts)

        # Strip ordinary trailing whitespace (spaces and tabs)
        if numTrailingWS > 0:
            stripped = raw[0:-numTrailingWS]
        else:
            stripped = raw
        return stripped

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def _parse_list(self) -> list:
        """
        Parse list items until ']'. Called after consuming '['.
        Each item is a tuple.
        """
        items: list = []

        while True:
            self.skip_blank_lines()

            if self.peek() == "]":
                self.advance()
                return items

            if self.peek() == "":
                raise self.error("unexpected EOF inside list")

            item = self._parse_tuple()
            items.append(item)

            self.skip_ws_inline()
            self.skip_comment()
            if self.peek() == "":
                raise self.error("unexpected EOF inside list")
            elif self.peek() != "\n":
                raise self.error(f"expected newline or ']', got {self.peek()!r}")

    # ------------------------------------------------------------------
    # Tuple
    # ------------------------------------------------------------------

    def _parse_tuple(self) -> tuple:
        """
        Parse space-separated values on the current logical line.
        '\\' at EOL continues on the next line.
        """
        values: list = []

        while True:
            self.skip_ws_inline()
            ch = self.peek()

            # End of logical line.
            if ch in ("\n", "#", "", "]", "}"):
                break

            values.append(self._parse_value())

        return tuple(values)

    # ------------------------------------------------------------------
    # Value dispatch
    # ------------------------------------------------------------------

    def _parse_value(self) -> object:
        """Dispatch to the correct value parser based on the current char."""
        ch = self.peek()

        # Quoted strings - check for raw (quote + newline) vs normal.
        if ch in ('"', "'"):
            if self.peek(1) == "\n":
                return self._parse_raw_string()
            return self._parse_normal_string()

        # Dictionary literal.
        if ch == "{":
            self.advance()
            return self._parse_dict_body("}")

        # List literal.
        if ch == "[":
            self.advance()
            return self._parse_list()

        # Number: starts with digit, or sign followed by digit.
        if ch.isdigit():
            return self._parse_number()

        if ch in ("+", "-") and self.peek(1).isdigit():
            return self._parse_number()

        # Simple string / keyword.
        if ch not in _STRUCTURAL and ch != "":
            return self._parse_simple_string()

        raise self.error(f"unexpected character {ch!r} in value position")

    # ------------------------------------------------------------------
    # Number
    # ------------------------------------------------------------------

    def _parse_number(self) -> Number:
        """
        Parse a MOMEL number: optional sign, optional base prefix,
        significand digits (with _ separators and optional decimal point),
        optional scientific exponent (_+N or _-N), optional unit suffix.
        """

        sig_parts: list[str] = []
        has_dot = False
        decimals = 0
        exponent_str = ""

        # 1. Optional sign.
        if self.peek() in _SIGN_CHARS:
            sig_parts.append(self.advance())

        # 2. Base prefix.
        base = 10
        base_digits: set[str]
        if self.peek() == "0":
            if self.peek(1) in _HEX_MARK:
                self.advance()
                self.advance()
                base = 16
                base_digits = _HEX_DIGITS
            elif self.peek(1) in _BIN_MARK:
                self.advance()
                self.advance()
                base = 2
                base_digits = _BIN_DIGITS
            elif self.peek(1) in _OCT_MARK:
                self.advance()
                self.advance()
                base = 8
                base_digits = _OCT_DIGITS
            else:
                base_digits = _DEC_DIGITS
        else:
            base_digits = _DEC_DIGITS

        # 3. Consume significand digits.
        #    State: we may or may not encounter a decimal point (decimal only)
        #    and we stop when we hit the exponent marker (_+/_-) or a suffix char.

        while True:
            ch = self.peek()

            if ch in base_digits:
                if has_dot:
                    decimals += 1

                sig_parts.append(ch)
                self.advance()
            elif ch == "_":
                # Peek at what follows the underscore.
                nxt = self.peek(1)
                if nxt in _SIGN_CHARS:
                    # Scientific notation exponent marker.
                    self.advance()  # consume '_'

                    exponent_str += self.advance()  # consume '+' or '-'

                    # Exponent digits are always decimal.
                    exponent_digits = ""
                    while self.peek().isdigit() or self.peek() == "_":
                        c = self.advance()
                        if c != "_":
                            exponent_digits += c

                    if not exponent_digits:
                        raise self.error("expected digits after exponent marker")

                    exponent_str += exponent_digits
                    break  # suffix may follow
                elif nxt in base_digits:
                    # Digit separator - skip the underscore.
                    self.advance()
                else:
                    # Underscore not followed by digit or sign -> start of suffix.
                    break
            elif ch == "." and not has_dot:
                # Decimal point, only once.
                has_dot = True
                self.advance()
            else:
                # Anything else ends the digit portion.
                break

        # 4. Build numeric value from significand.
        numeric_val: int | float

        sig_str = "".join(sig_parts)
        if not sig_str:
            raise self.error("expected digits in number")

        if has_dot or exponent_str != "":
            # Float path.
            if base == 10 and (
                has_dot or (exponent_str != "" and exponent_str[0] == "-")
            ):
                if decimals > 0:
                    py_num_str = f"{sig_str[0:-decimals]}.{sig_str[-decimals:]}"
                else:
                    py_num_str = sig_str

                if exponent_str != "":
                    py_num_str += f"e{exponent_str}"

                try:
                    numeric_val = float(py_num_str)
                except ValueError:
                    raise self.error(f"invalid number: {py_num_str!r}")
            else:
                try:
                    significand_val = int(sig_str, base)
                except ValueError:
                    raise self.error(f"invalid number significand: {sig_str!r}")

                power_val = -decimals
                if exponent_str != "":
                    try:
                        exponent_val = int(exponent_str, 10)
                        power_val += exponent_val
                    except ValueError:
                        raise self.error(f"invalid number exponent: {exponent_str!r}")

                numeric_val = significand_val * (base**power_val)
        else:
            try:
                numeric_val = int(sig_str, base)
            except ValueError:
                raise self.error(f"invalid number: {sig_str!r}")

        # 7. Parse optional unit suffix.
        unit = self._parse_suffix()

        return Number(numeric_val, unit)

    def _parse_suffix(self) -> str:
        """Consume suffix chars (is_suffix_char or backslash-escaped)."""
        parts: list[str] = []
        while True:
            ch = self.peek()
            if ch == "\\":
                self.advance()  # consume '\'
                parts.append(self.advance_escape())
            elif is_suffix_char(ch):
                parts.append(ch)
                self.advance()
            else:
                break
        return "".join(parts)

    # ------------------------------------------------------------------
    # Strings
    # ------------------------------------------------------------------

    def _parse_normal_string(self) -> str:
        """Parse a single- or double-quoted string with escape sequences."""
        q = self.advance()  # opening quote char
        parts: list[str] = []

        while True:
            ch = self.peek()

            if ch == "":
                raise self.error("unexpected EOF inside string")

            if ch == "\n":
                raise self.error(
                    "unexpected newline inside normal string (use raw string)"
                )

            if ch == q:
                self.advance()  # closing quote
                break

            if ch == "\\":
                self.advance()  # consume '\'
                parts.append(self.advance_escape())
            else:
                parts.append(ch)
                self.advance()

        return "".join(parts)

    def _parse_raw_string(self) -> str:
        """
        Parse a raw (multiline) string.
        Called when the opening quote is immediately followed by '\\n'.

        Indentation rules:
          - opening quote is at column open_col (0-based)
          - each content line must be indented by (open_col + 1) spaces
          - closing quote must start at column open_col
          - no escape sequences; content is literal
        """
        # Capture column before consuming the opening quote.
        open_col = self.current_col()
        q = self.advance()  # consume opening quote
        self.advance()  # consume '\n' that immediately follows

        lines: list[str] = []

        while True:
            # Consume `indent` leading spaces.
            # Empty/whitespace-only lines (fewer than indent spaces) are allowed
            # and produce an empty string in the output.
            is_empty_line = False
            for _ in range(open_col):
                if self.peek() == " ":
                    self.advance()
                elif self.peek() == "\n":
                    is_empty_line = True
                    break
                elif self.peek() == "":
                    raise self.error(
                        f"Unexpected EOF inside raw string (before content line or closing quote)"
                    )
                else:
                    raise self.error(
                        f"raw string content line must be indented by {open_col+1} spaces"
                    )

            if self.peek() == "":
                raise self.error(
                    f"Unexpected EOF inside raw string (before content line or closing quote)"
                )

            if is_empty_line:
                lines.append("")
                self.advance()  # consume '\n'
                continue

            # Check the first character after the indent - could be a quote, a space followed by a line, or an error
            c = self.advance()
            if c == q:
                # Closing quote line
                return "\n".join(lines)
            elif c == "\n":
                # Empty lines could also have the end of line immediately after the indent
                lines.append("")
            elif c == " ":
                # Otherwise this is a content line
                # Collect rest of line up to (not including) '\n'.
                line_parts: list[str] = []
                while True:
                    if self.peek() == "\n":
                        lines.append("".join(line_parts))
                        self.advance()
                        break
                    elif self.peek() == "":
                        raise self.error(
                            "unexpected EOF inside raw string (missing closing quote)"
                        )
                    else:
                        line_parts.append(self.advance())

    def _parse_simple_string(self) -> str:
        """
        Parse a simple (unquoted) string / keyword.
        Same character class as number suffixes.
        Starts with a non-digit (enforced by caller via dispatch).
        """
        return self._parse_suffix()


# ---------------------------------------------------------------------------
# Dict merge utility
# ---------------------------------------------------------------------------


def _merge_dicts(a: dict, b: dict) -> dict:
    """
    Recursively merge dict b into dict a.
    Duplicate keys that are both (dict,) tuples are merged recursively.
    Other duplicates raise ParseError (no location - caught earlier).
    """
    result = dict(a)
    for key, b_val in b.items():
        if key in result:
            a_val = result[key]
            if (
                len(a_val) == 1
                and isinstance(a_val[0], dict)
                and len(b_val) == 1
                and isinstance(b_val[0], dict)
            ):
                result[key] = (_merge_dicts(a_val[0], b_val[0]),)
            else:
                raise ParseError(f"duplicate key {key!r} in merged dicts", 0, 0)
        else:
            result[key] = b_val
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(text: str) -> dict:
    """Parse a MOMEL string and return the top-level dictionary."""
    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _Parser(text).parse_top_level()


def parse_file(path: str | os.PathLike) -> dict:
    """Read a MOMEL file (UTF-8) and return the top-level dictionary."""
    with open(path, encoding="utf-8") as f:
        return parse(f.read())

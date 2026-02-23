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
        if len(hex_str) < 2 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
            raise ParseError(r"expected 2 hex digits after \x", line, col)
        return chr(int(hex_str, 16)), pos + 3

    # \uHHHH
    if ch == "u":
        hex_str = src[pos + 1 : pos + 5]
        if len(hex_str) < 4 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
            raise ParseError(r"expected 4 hex digits after \u", line, col)
        return chr(int(hex_str, 16)), pos + 5

    # \UHHHHHHHH
    if ch == "U":
        hex_str = src[pos + 1 : pos + 9]
        if len(hex_str) < 8 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
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
        while self.peek() in (" ", "\t"):
            self.advance()

    def skip_comment(self) -> None:
        """If at '#', advance to (not past) the newline/EOF."""
        if self.peek() == "#":
            while self.peek() not in ("\n", ""):
                self.advance()

    def skip_blank_lines(self) -> None:
        """Skip lines that are entirely whitespace or comments."""
        while True:
            # Save position in case this line isn't blank.
            saved_pos = self.pos
            saved_line = self.line
            saved_ls = self.line_start

            self.skip_ws_inline()
            self.skip_comment()

            if self.peek() == "\n":
                self.advance()  # consume the newline and keep going
            elif self.peek() == "":
                break  # EOF is fine
            else:
                # Non-blank line: restore and stop.
                self.pos = saved_pos
                self.line = saved_line
                self.line_start = saved_ls
                break

    def at_eol(self) -> bool:
        """True if current position is at end of logical line (newline, EOF, #, ], })."""
        ch = self.peek()
        return ch in ("\n", "", "#", "]", "}")

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
            # Skip indentation before the field identifier or closing brace.
            self.skip_ws_inline()

            ch = self.peek()

            # End of dict body.
            if closing is not None and ch == closing:
                self.advance()
                return result

            # EOF while parsing top-level dict is fine.
            if ch == "":
                if closing is not None:
                    raise self.error(f"unexpected EOF, expected {closing!r}")
                return result

            # Parse one field.
            key = self._parse_identifier()
            self.skip_ws_inline()
            self.expect(":")
            self.skip_ws_inline()
            val = self._parse_tuple()

            # Expect end of line (or '}' immediately closing the dict).
            self.skip_ws_inline()
            self.skip_comment()
            if self.peek() not in ("\n", "", "}" if closing == "}" else "\n"):
                # Allow closing brace to follow on same line as last field.
                if closing == "}" and self.peek() == "}":
                    pass
                else:
                    raise self.error(f"expected end of line, got {self.peek()!r}")
            if self.peek() == "\n":
                self.advance()

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

        # Unreachable but satisfies type checkers.
        return result  # pragma: no cover

    def _parse_identifier(self) -> str:
        """
        Consume chars until unescaped ':' or newline.
        Apply escape sequences; strip leading/trailing ASCII whitespace.
        \\_ escape -> literal space (preserved, not stripped).
        """
        parts: list[str] = []

        while True:
            ch = self.peek()

            if ch == "" or ch == "\n":
                break

            if ch == ":":
                break

            if ch == "\\":
                # Line continuation inside identifier: treat like tuple continuation.
                if self.peek(1) == "\n":
                    self.advance()  # consume '\'
                    self.advance()  # consume '\n'
                    continue

                self.advance()  # consume '\'
                decoded, new_pos = decode_escape(
                    self.src, self.pos, self.line, self.current_col()
                )
                # \_ is a non-strippable space — mark it so stripping won't remove it.
                # We encode it as a special sentinel during accumulation; see below.
                if self.src[self.pos - 1 : self.pos] == "" and decoded == " ":
                    # Actually we already consumed '\', pos now points to '_'
                    pass
                # Check if the escape was \_  (literal space that should not be trimmed).
                # We handle this by looking at what escape char was.
                esc_char = self.src[self.pos] if self.pos < len(self.src) else ""
                self.pos = new_pos
                if esc_char == "_":
                    # Non-strippable space: use a private sentinel.
                    parts.append("\x00NBSP\x00")
                else:
                    parts.append(decoded)
                continue

            if is_identifier_char(ch):
                parts.append(ch)
                self.advance()
            else:
                # Structural char that's not ':' or newline.
                parts.append(ch)
                self.advance()

        raw = "".join(parts)

        # Strip ordinary leading/trailing whitespace (spaces and tabs), but
        # restore \_ sentinels to actual spaces.
        stripped = raw.strip(" \t")
        result = stripped.replace("\x00NBSP\x00", " ")
        return result

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
            self.skip_ws_inline()

            if self.peek() == "]":
                self.advance()
                return items

            if self.peek() == "":
                raise self.error("unexpected EOF inside list")

            item = self._parse_tuple()
            items.append(item)

            self.skip_ws_inline()
            self.skip_comment()
            if self.peek() == "\n":
                self.advance()
            elif self.peek() == "]":
                pass  # closing bracket follows immediately
            elif self.peek() == "":
                raise self.error("unexpected EOF inside list")
            else:
                raise self.error(f"expected newline or ']', got {self.peek()!r}")

        return items  # pragma: no cover

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
            if ch in ("\n", "", "]", "}"):
                break

            # Comment → end of tuple.
            if ch == "#":
                self.skip_comment()
                break

            # Line continuation: '\' followed immediately by '\n'.
            if ch == "\\":
                next_ch = self.peek(1)
                if next_ch == "\n":
                    self.advance()  # consume '\'
                    self.advance()  # consume '\n'
                    # Skip any leading whitespace on the continuation line.
                    self.skip_ws_inline()
                    continue
                # Not a continuation - fall through to parse_value
                # (backslash starts an escape in a simple string or identifier context,
                #  but a bare '\' at the value-start position is unusual;
                #  parse_value will handle it via simple-string path).

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

        # Hex/binary/octal prefix: 0x, 0b, 0o.
        if ch == "0" and self.peek(1) in ("x", "b", "o"):
            return self._parse_number()

        # Simple string / keyword.
        if is_suffix_char(ch) or ch == "\\":
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
        # 1. Optional sign.
        sign = 1
        if self.peek() in ("+", "-"):
            sign = -1 if self.advance() == "-" else 1

        # 2. Base prefix.
        base = 10
        base_digits: set[str]
        if self.peek() == "0" and self.peek(1) in ("x", "X"):
            self.advance()
            self.advance()
            base = 16
            base_digits = set("0123456789abcdefABCDEF")
        elif self.peek() == "0" and self.peek(1) in ("b", "B"):
            self.advance()
            self.advance()
            base = 2
            base_digits = set("01")
        elif self.peek() == "0" and self.peek(1) in ("o", "O"):
            self.advance()
            self.advance()
            base = 8
            base_digits = set("01234567")
        else:
            base_digits = set("0123456789")

        # 3. Consume significand digits.
        #    State: we may or may not encounter a decimal point (decimal only)
        #    and we stop when we hit the exponent marker (_+/_-) or a suffix char.
        sig_parts: list[str] = []
        has_dot = False
        exponent_sign = 1
        exponent_digits = ""
        has_exponent = False

        while True:
            ch = self.peek()

            if ch in base_digits:
                sig_parts.append(ch)
                self.advance()

            elif ch == "_":
                # Peek at what follows the underscore.
                nxt = self.peek(1)
                if nxt in ("+", "-"):
                    # Scientific notation exponent marker.
                    self.advance()  # consume '_'
                    exp_sign_ch = self.advance()  # consume '+' or '-'
                    exponent_sign = -1 if exp_sign_ch == "-" else 1
                    # Exponent digits are always decimal.
                    exp_str_parts: list[str] = []
                    while self.peek().isdigit() or self.peek() == "_":
                        c = self.advance()
                        if c != "_":
                            exp_str_parts.append(c)
                    exponent_digits = "".join(exp_str_parts)
                    if not exponent_digits:
                        raise self.error("expected digits after exponent marker")
                    has_exponent = True
                    break  # suffix may follow
                elif nxt in base_digits:
                    # Digit separator - skip the underscore.
                    self.advance()
                else:
                    # Underscore not followed by digit or sign -> start of suffix.
                    break

            elif ch == "." and base == 10 and not has_dot:
                # Decimal point - only in decimal mode, only once.
                has_dot = True
                sig_parts.append(ch)
                self.advance()

            else:
                # Anything else ends the digit portion.
                break

        # 4. Build numeric value from significand.
        sig_str = "".join(sig_parts).replace("_", "")
        if not sig_str:
            raise self.error("expected digits in number")

        if has_dot or (has_exponent and exponent_sign < 0):
            # Float path.
            try:
                if base == 10:
                    numeric_val: int | float = float(sig_str)
                else:
                    # Non-decimal float: parse integer part then convert.
                    numeric_val = float(int(sig_str, base))
            except ValueError:
                raise self.error(f"invalid number significand: {sig_str!r}")
        else:
            try:
                numeric_val = int(sig_str, base)
            except ValueError:
                raise self.error(f"invalid number significand: {sig_str!r}")

        # 5. Apply sign.
        numeric_val = sign * numeric_val

        # 6. Apply exponent: value *= base ** (sign * exp).
        if has_exponent:
            exp_val = int(exponent_digits) * exponent_sign
            if base == 10:
                # Delegate to Python's float parser to avoid precision loss
                # from manual multiplication (e.g. 6.02 * 10**23).
                exp_str = f"{'+' if exp_val >= 0 else ''}{exp_val}"
                numeric_val = float(f"{sig_str}e{exp_str}") * sign
                # Keep as int when result is a whole number with no '.'
                if not has_dot and exp_val >= 0:
                    int_val = int(numeric_val)
                    if int_val == numeric_val:
                        numeric_val = int_val
            else:
                # Non-decimal base: multiply by base^exponent.
                if exp_val < 0:
                    numeric_val = float(numeric_val) * (base**exp_val)
                else:
                    factor = base**exp_val
                    numeric_val = numeric_val * factor
                    # Keep as int if no fractional component.
                    if isinstance(numeric_val, float) and not has_dot:
                        int_val = int(numeric_val)
                        if int_val == numeric_val:
                            numeric_val = int_val

        # 7. Parse optional unit suffix.
        unit = self._parse_suffix()

        return Number(numeric_val, unit)

    def _parse_suffix(self) -> str:
        """Consume suffix chars (is_suffix_char or backslash-escaped)."""
        parts: list[str] = []
        while True:
            ch = self.peek()
            if ch == "\\":
                if self.peek(1) == "\n":
                    break  # line continuation — not part of suffix
                self.advance()  # consume '\'
                decoded, new_pos = decode_escape(
                    self.src, self.pos, self.line, self.current_col()
                )
                self.pos = new_pos
                parts.append(decoded)
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
                decoded, new_pos = decode_escape(
                    self.src, self.pos, self.line, self.current_col()
                )
                self.pos = new_pos
                parts.append(decoded)
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
        indent = open_col + 1
        q = self.advance()  # consume opening quote
        self.advance()  # consume '\n' that immediately follows

        lines: list[str] = []

        while True:
            if self.peek() == "":
                raise self.error(
                    "unexpected EOF inside raw string (missing closing quote)"
                )

            # Check for closing quote line: exactly open_col spaces then q.
            # We peek ahead to decide.
            closing_prefix = self.src[self.pos : self.pos + open_col + 1]
            if closing_prefix == " " * open_col + q:
                # Advance past the closing quote.
                for _ in range(open_col + 1):
                    self.advance()
                return "\n".join(lines)

            # Otherwise this is a content line — consume `indent` leading spaces.
            # Empty/whitespace-only lines (fewer than indent spaces) are allowed
            # and produce an empty string in the output.
            is_empty_line = False
            for _ in range(indent):
                if self.peek() == " ":
                    self.advance()
                elif self.peek() == "\n":
                    is_empty_line = True
                    break
                else:
                    raise self.error(
                        f"raw string content line must be indented by {indent} spaces"
                    )

            if is_empty_line:
                lines.append("")
                self.advance()  # consume '\n'
                continue

            # Collect rest of line up to (not including) '\n'.
            line_parts: list[str] = []
            while self.peek() not in ("\n", ""):
                line_parts.append(self.advance())

            lines.append("".join(line_parts))

            if self.peek() == "\n":
                self.advance()
            else:
                raise self.error("unexpected EOF inside raw string")

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

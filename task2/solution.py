"""
Context Collection Pipeline for EnsembleAI 2026 Task 2.

Multi-signal retrieval: AST import resolution + symbol extraction +
cursor-weighted BM25 + 3-zone token-budget composition.
"""

import ast
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Optional

import jsonlines
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILE_SEP = "<|file_sep|>"
CHARS_PER_TOKEN = 3.8  # conservative estimate
MELLUM_TOKENS = 8_000
FULL_TOKENS = 16_000
TARGET_CHARS = int(FULL_TOKENS * CHARS_PER_TOKEN)
MELLUM_CHARS = int(MELLUM_TOKENS * CHARS_PER_TOKEN)

CORE_CHARS = int(6_500 * CHARS_PER_TOKEN)
EXTENDED_CHARS = int(14_000 * CHARS_PER_TOKEN)

PYTHON_STDLIB = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
    "cmath", "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "formatter",
    "fractions", "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
    "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd",
    "smtplib", "sndhdr", "socket", "socketserver", "sqlite3", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu", "uuid",
    "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
    "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib", "_thread", "__future__",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Context collection pipeline")
    p.add_argument("--stage", default="start")
    p.add_argument("--lang", default="python")
    p.add_argument("--trim-prefix-lines", type=int, default=0,
                    help="If >0, keep only the last N lines of the prefix")
    p.add_argument("--trim-suffix-lines", type=int, default=0,
                    help="If >0, keep only the first N lines of the suffix")
    p.add_argument("--bm25-top-k", type=int, default=5,
                    help="Number of BM25 candidate files")
    p.add_argument("--max-context-chars", type=int, default=EXTENDED_CHARS,
                    help="Max total context characters")
    p.add_argument("--core-chars", type=int, default=CORE_CHARS,
                    help="Max chars for core zone (rightmost, Mellum-safe)")
    p.add_argument("--mode", default="dual",
                    choices=["precision", "full", "dual"],
                    help="dual = two-zone; precision = compact; full = 16K")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Repository scanner – build a virtual filesystem
# ---------------------------------------------------------------------------

def scan_repo(root_dir: str, ext: str = ".py") -> dict[str, str]:
    """Return {relative_path: file_content} for all matching files."""
    files: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(ext):
                continue
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, root_dir)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    files[rel_path] = f.read()
            except Exception:
                pass
    return files

# ---------------------------------------------------------------------------
# AST Pass 1 – Import extraction
# ---------------------------------------------------------------------------

def _safe_parse(source: str) -> Optional[ast.Module]:
    """Parse Python source, handling partial files (prefix cut at cursor)."""
    try:
        return ast.parse(source)
    except SyntaxError:
        pass

    # Partial file: try appending 'pass' to close any open blocks
    try:
        return ast.parse(source + "\n    pass\n")
    except SyntaxError:
        pass

    # Still failing: try parsing only import lines at the top
    lines = source.split("\n")
    safe_lines = []
    for line in lines:
        stripped = line.strip()
        if (not stripped or stripped.startswith("#")
                or stripped.startswith("import ")
                or stripped.startswith("from ")):
            safe_lines.append(line)
        elif stripped.startswith(("class ", "def ", "async def ")):
            break
        else:
            safe_lines.append(line)

    try:
        return ast.parse("\n".join(safe_lines))
    except SyntaxError:
        return None


def extract_imports(source: str) -> list[dict]:
    """Extract import statements from Python source.

    Returns list of dicts with keys:
      - module: dotted module path
      - names: list of imported names (for 'from X import a,b')
      - is_relative: bool
      - level: relative import level (0 = absolute)
    """
    tree = _safe_parse(source)
    if tree is None:
        return _extract_imports_regex(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "names": [],
                    "is_relative": False,
                    "level": 0,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append({
                "module": module,
                "names": [a.name for a in node.names],
                "is_relative": node.level > 0,
                "level": node.level,
            })
    return imports


def _extract_imports_regex(source: str) -> list[dict]:
    """Fallback: extract imports via regex when AST parsing fails entirely."""
    imports = []
    for line in source.split("\n"):
        line = line.strip()
        m = re.match(r"^from\s+(\.*)(\S*)\s+import\s+(.+)", line)
        if m:
            dots, module, names_str = m.groups()
            level = len(dots)
            names = [n.strip().split(" as ")[0] for n in names_str.split(",")]
            imports.append({
                "module": module,
                "names": names,
                "is_relative": level > 0,
                "level": level,
            })
            continue
        m = re.match(r"^import\s+(.+)", line)
        if m:
            for part in m.group(1).split(","):
                name = part.strip().split(" as ")[0].strip()
                imports.append({
                    "module": name,
                    "names": [],
                    "is_relative": False,
                    "level": 0,
                })
    return imports


def is_stdlib(module: str) -> bool:
    top = module.split(".")[0]
    return top in PYTHON_STDLIB


def resolve_import(imp: dict, active_file_path: str, repo_files: dict[str, str]) -> Optional[str]:
    """Resolve an import dict to a relative file path within the repo.

    Returns the matched relative path or None.
    """
    module = imp["module"]

    if not imp["is_relative"] and is_stdlib(module):
        return None

    if imp["is_relative"]:
        active_dir = os.path.dirname(active_file_path)
        level = imp["level"]
        base = active_dir
        for _ in range(level - 1):
            base = os.path.dirname(base)
        if module:
            parts = module.split(".")
            dotted_path = os.path.join(base, *parts)
        else:
            dotted_path = base
    else:
        parts = module.split(".")
        dotted_path = os.path.join(*parts) if len(parts) > 1 else parts[0]

    candidates = [
        dotted_path + ".py",
        os.path.join(dotted_path, "__init__.py"),
    ]

    for c in candidates:
        norm = os.path.normpath(c)
        if norm in repo_files:
            return norm

    # Try partial matches (package prefix)
    for c in candidates:
        norm = os.path.normpath(c)
        for rp in repo_files:
            if rp.endswith(norm) or rp.endswith(os.sep + norm):
                return rp

    return None

# ---------------------------------------------------------------------------
# AST Pass 2 – Symbol usage extraction
# ---------------------------------------------------------------------------

def extract_used_symbols(source: str) -> set[str]:
    """Extract names of called functions, instantiated classes, and referenced
    attributes from the source (prefix + suffix around cursor)."""
    tree = _safe_parse(source)
    if tree is None:
        return _extract_symbols_regex(source)

    symbols = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                symbols.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                symbols.add(node.func.attr)
    return symbols


def _extract_symbols_regex(source: str) -> set[str]:
    """Fallback symbol extraction via regex."""
    return set(_TOKEN_RE.findall(source))


def extract_type_hints(source: str) -> set[str]:
    """Extract type annotation names from function signatures and assignments."""
    tree = _safe_parse(source)
    if tree is None:
        return set()

    hints = set()

    def _collect_annotation(ann):
        if isinstance(ann, ast.Name):
            hints.add(ann.id)
        elif isinstance(ann, ast.Attribute):
            hints.add(ann.attr)
        elif isinstance(ann, ast.Subscript):
            _collect_annotation(ann.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation:
                    _collect_annotation(arg.annotation)
            if node.returns:
                _collect_annotation(node.returns)
        elif isinstance(node, ast.AnnAssign) and node.annotation:
            _collect_annotation(node.annotation)

    return hints


def extract_base_classes(source: str) -> set[str]:
    """Extract base class names from class definitions in the source."""
    tree = _safe_parse(source)
    if tree is None:
        return set()

    bases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.add(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.add(base.attr)

    bases -= {"object", "Exception", "BaseException", "type",
              "dict", "list", "set", "tuple", "str", "int", "float", "bool"}
    return bases

# ---------------------------------------------------------------------------
# AST Pass 3 – Targeted extraction from candidate files
# ---------------------------------------------------------------------------

def extract_definitions(source: str) -> list[dict]:
    """Parse a file and return its top-level definitions.

    Each dict: {name, kind, start_line, end_line, source}
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    defs = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append(_def_dict(node, lines, "function"))
        elif isinstance(node, ast.ClassDef):
            defs.append(_def_dict(node, lines, "class"))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs.append(_def_dict(node, lines, "assignment", target.id))

    return defs


def _def_dict(node, lines, kind, name=None):
    name = name or getattr(node, "name", "?")
    start = node.lineno - 1
    end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
    src = "".join(lines[start:end])
    return {"name": name, "kind": kind, "start": start, "end": end, "source": src}


def extract_relevant_snippets(file_source: str, needed_symbols: set[str],
                               fallback_skeleton: bool = True) -> str:
    """From a file, extract only the definitions that match needed symbols.

    If none match and fallback_skeleton is True, return a skeleton.
    If fallback_skeleton is False, return empty string.
    """
    defs = extract_definitions(file_source)
    if not defs:
        return skeletonize(file_source) if fallback_skeleton else ""

    matched = [d for d in defs if d["name"] in needed_symbols]

    if not matched:
        return skeletonize(file_source) if fallback_skeleton else ""

    return "\n\n".join(d["source"] for d in matched)


def skeletonize(file_source: str) -> str:
    """Create a skeleton: class/function signatures with 'pass', stripping
    docstrings and implementation bodies. Falls back to full source on parse error."""
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        return file_source

    lines = file_source.splitlines(keepends=True)
    skeleton_parts = []

    # Keep top-level imports and assignments
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            skeleton_parts.append("".join(lines[start:end]))
        elif isinstance(node, ast.ClassDef):
            skeleton_parts.append(_skeletonize_class(node, lines))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            skeleton_parts.append(_skeletonize_func(node, lines))
        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            skeleton_parts.append("".join(lines[start:end]))

    result = "\n".join(skeleton_parts)
    return result if result.strip() else file_source


def _skeletonize_func(node, lines):
    sig_line = lines[node.lineno - 1]
    # Find the colon to get just the signature
    return sig_line.rstrip() + "\n" + " " * (node.col_offset + 4) + "...\n"


def _skeletonize_class(node, lines):
    parts = [lines[node.lineno - 1].rstrip()]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            indent = " " * child.col_offset
            sig = lines[child.lineno - 1].rstrip()
            parts.append(sig)
            parts.append(indent + "    ...")
        elif isinstance(child, ast.Assign):
            start = child.lineno - 1
            end = child.end_lineno if hasattr(child, "end_lineno") and child.end_lineno else start + 1
            parts.append("".join(lines[start:end]).rstrip())
    return "\n".join(parts) + "\n"

# ---------------------------------------------------------------------------
# Strip docstrings from external files
# ---------------------------------------------------------------------------

def strip_docstrings(source: str) -> str:
    """Remove module/function/class docstrings to save tokens."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    ranges_to_remove = []

    def _check_docstring(node):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)):
            ds = node.body[0]
            start = ds.lineno - 1
            end = ds.end_lineno if hasattr(ds, "end_lineno") and ds.end_lineno else start + 1
            ranges_to_remove.append((start, end))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check_docstring(node)

    if not ranges_to_remove:
        return source

    remove_set = set()
    for start, end in ranges_to_remove:
        for i in range(start, end):
            remove_set.add(i)

    return "".join(line for i, line in enumerate(lines) if i not in remove_set)

# ---------------------------------------------------------------------------
# BM25 – cursor-weighted
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z_]\w*")

def tokenize_code(text: str) -> list[str]:
    """Split code into identifier-level tokens."""
    return _TOKEN_RE.findall(text.lower())


def cursor_weighted_query(prefix: str, suffix: str, cursor_chars: int = 500) -> list[str]:
    """Build a BM25 query emphasizing tokens near the cursor."""
    near_prefix = prefix[-cursor_chars:] if len(prefix) > cursor_chars else prefix
    near_suffix = suffix[:cursor_chars] if len(suffix) > cursor_chars else suffix
    far_prefix = prefix[:-cursor_chars] if len(prefix) > cursor_chars else ""

    near_tokens = tokenize_code(near_prefix + " " + near_suffix)
    far_tokens = tokenize_code(far_prefix)

    # Weight near-cursor tokens 3x
    return near_tokens * 3 + far_tokens


class NativeBM25:
    """Lightweight BM25 implementation using only stdlib."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.n if self.n else 1

        self.df: Counter = Counter()
        self.doc_lens = []
        self.doc_freqs = []

        for doc in corpus:
            self.doc_lens.append(len(doc))
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for term in freq:
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query:
            idf = self._idf(term)
            for i in range(self.n):
                tf = self.doc_freqs[i].get(term, 0)
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom if denom else 0
        return scores

# ---------------------------------------------------------------------------
# Context composer – 3-zone relevance gradient
# ---------------------------------------------------------------------------

def compose_context(
    zone_right: list[tuple[str, str]],     # (rel_path, content) – highest priority
    zone_middle: list[tuple[str, str]],     # imported files / siblings
    zone_left: list[tuple[str, str]],       # BM25 matches
    max_chars: int = TARGET_CHARS,
) -> str:
    """Assemble context string with relevance gradient.

    Layout: [zone_left ... zone_middle ... zone_right]
    Left-trimming by the eval harness removes zone_left first.
    """

    def _format_file(path: str, content: str) -> str:
        return f"{FILE_SEP}{path}\n{content}"

    # Build from right to left, respecting budget
    parts_right = []
    used = 0
    for path, content in zone_right:
        entry = _format_file(path, content)
        if used + len(entry) > max_chars:
            remaining = max_chars - used
            if remaining > 100:
                parts_right.append(entry[:remaining])
                used = max_chars
            break
        parts_right.append(entry)
        used += len(entry)

    parts_middle = []
    for path, content in zone_middle:
        entry = _format_file(path, content)
        if used + len(entry) > max_chars:
            remaining = max_chars - used
            if remaining > 100:
                parts_middle.append(entry[:remaining])
                used = max_chars
            break
        parts_middle.append(entry)
        used += len(entry)

    parts_left = []
    for path, content in zone_left:
        entry = _format_file(path, content)
        if used + len(entry) > max_chars:
            remaining = max_chars - used
            if remaining > 100:
                parts_left.append(entry[:remaining])
                used = max_chars
            break
        parts_left.append(entry)
        used += len(entry)

    # Assemble: left (least important) -> middle -> right (most important)
    all_parts = list(reversed(parts_left)) + list(reversed(parts_middle)) + list(reversed(parts_right))
    return "".join(all_parts)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_datapoint(
    datapoint: dict,
    repo_files: dict[str, str],
    active_path: str,
    args,
) -> dict:
    """Process a single datapoint and return the submission dict."""

    prefix = datapoint["prefix"]
    suffix = datapoint.get("suffix", "")
    modified = datapoint.get("modified", [])

    core_budget = getattr(args, "core_chars", CORE_CHARS)
    total_budget = args.max_context_chars

    candidate_files = {p: c for p, c in repo_files.items() if p != active_path}

    # --- AST Pass 1: Import extraction & resolution ---
    imports = extract_imports(prefix)
    imported_paths: list[str] = []
    for imp in imports:
        resolved = resolve_import(imp, active_path, repo_files)
        if resolved and resolved != active_path and resolved in candidate_files:
            if resolved not in imported_paths:
                imported_paths.append(resolved)

    if suffix:
        suffix_imports = extract_imports(suffix)
        for imp in suffix_imports:
            resolved = resolve_import(imp, active_path, repo_files)
            if resolved and resolved != active_path and resolved in candidate_files:
                if resolved not in imported_paths:
                    imported_paths.append(resolved)

    # --- AST Pass 2: Symbols + type hints + base classes ---
    used_symbols = extract_used_symbols(prefix)
    if suffix:
        used_symbols |= extract_used_symbols(suffix)
    type_hints = extract_type_hints(prefix)
    base_classes = extract_base_classes(prefix)
    if suffix:
        base_classes |= extract_base_classes(suffix)
    needed_symbols = used_symbols | type_hints | base_classes

    # ===== CORE ZONE (rightmost — all 3 models see this) =====
    core_entries: list[tuple[str, str]] = []
    core_used = 0
    included_paths: set[str] = set()

    for imp_path in imported_paths:
        if core_used >= core_budget:
            break

        src = candidate_files[imp_path]
        if imp_path.endswith("__init__.py"):
            defs = extract_definitions(src)
            if not defs or len(src) < 50:
                continue

        snippets = extract_relevant_snippets(src, needed_symbols, fallback_skeleton=True)
        snippets = strip_docstrings(snippets)
        if not snippets.strip():
            continue

        entry_len = len(FILE_SEP) + len(imp_path) + 1 + len(snippets)
        if core_used + entry_len > core_budget:
            remaining = core_budget - core_used
            if remaining > 200:
                skel = skeletonize(src)
                skel = strip_docstrings(skel)
                skel_len = len(FILE_SEP) + len(imp_path) + 1 + len(skel)
                if skel_len < remaining:
                    core_entries.append((imp_path, skel))
                    core_used += skel_len
                    included_paths.add(imp_path)
            break

        core_entries.append((imp_path, snippets))
        core_used += entry_len
        included_paths.add(imp_path)

    # Parent class + type hint resolution in core zone
    names_to_find = (type_hints | base_classes) - {s for s in (type_hints | base_classes) if len(s) <= 2}
    if names_to_find and core_used < core_budget:
        for path, content in candidate_files.items():
            if path in included_paths or core_used >= core_budget:
                break
            for tname in names_to_find:
                if f"class {tname}" in content:
                    snippets = extract_relevant_snippets(content, {tname}, fallback_skeleton=True)
                    snippets = strip_docstrings(snippets)
                    entry_len = len(FILE_SEP) + len(path) + 1 + len(snippets)
                    if core_used + entry_len <= core_budget:
                        core_entries.append((path, snippets))
                        included_paths.add(path)
                        core_used += entry_len
                    break

    # ===== EXTENDED ZONE (leftmost — only Codestral/Qwen see this) =====
    extended_entries: list[tuple[str, str]] = []
    extended_budget = total_budget - core_used
    extended_used = 0

    # Modified files — recently edited, very high relevance signal
    for mod_path in modified:
        if extended_used >= extended_budget:
            break
        if mod_path in included_paths or mod_path == active_path:
            continue
        if mod_path not in candidate_files:
            matched = None
            for rp in candidate_files:
                if rp.endswith(mod_path) or rp.endswith(os.sep + mod_path):
                    matched = rp
                    break
            if not matched:
                continue
            mod_path = matched

        src = candidate_files[mod_path]
        snippets = extract_relevant_snippets(src, needed_symbols, fallback_skeleton=True)
        snippets = strip_docstrings(snippets)
        if len(snippets.strip()) < 30:
            continue

        entry_len = len(FILE_SEP) + len(mod_path) + 1 + len(snippets)
        if extended_used + entry_len <= extended_budget:
            extended_entries.append((mod_path, snippets))
            included_paths.add(mod_path)
            extended_used += entry_len

    # BM25 in extended zone
    if extended_used < extended_budget and args.bm25_top_k > 0:
        remaining_files = {p: c for p, c in candidate_files.items()
                          if p not in included_paths and not p.endswith("__init__.py")}

        if remaining_files:
            query = cursor_weighted_query(prefix, suffix)
            file_paths = list(remaining_files.keys())
            corpus = [tokenize_code(remaining_files[p]) for p in file_paths]

            if corpus:
                bm25 = NativeBM25(corpus)
                scores = bm25.score(query)
                ranked = sorted(zip(scores, file_paths), reverse=True)

                for score_val, path in ranked[:args.bm25_top_k]:
                    if score_val <= 0 or extended_used >= extended_budget:
                        break
                    skeleton = skeletonize(remaining_files[path])
                    skeleton = strip_docstrings(skeleton)
                    if len(skeleton.strip()) < 50:
                        continue
                    entry_len = len(FILE_SEP) + len(path) + 1 + len(skeleton)
                    if extended_used + entry_len <= extended_budget:
                        extended_entries.append((path, skeleton))
                        extended_used += entry_len

    # ===== COMPOSE: extended (left, trimmed first) + core (right, survives) =====
    all_entries = extended_entries + core_entries
    context = "".join(f"{FILE_SEP}{path}\n{content}" for path, content in all_entries)

    submission: dict = {"context": context}

    if args.trim_prefix_lines > 0:
        plines = prefix.split("\n")
        if len(plines) > args.trim_prefix_lines:
            submission["prefix"] = "\n".join(plines[-args.trim_prefix_lines:])

    if args.trim_suffix_lines > 0 and suffix:
        slines = suffix.split("\n")
        if len(slines) > args.trim_suffix_lines:
            submission["suffix"] = "\n".join(slines[:args.trim_suffix_lines])

    return submission


def main():
    args = parse_args()
    stage = args.stage
    language = args.lang

    if language == "python":
        ext = ".py"
    elif language == "kotlin":
        ext = ".kt"
    else:
        raise ValueError(f"Unsupported language: {language}")

    input_file = os.path.join("data", f"{language}-{stage}.jsonl")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    # Count datapoints for progress bar
    with open(input_file) as f:
        total = sum(1 for _ in f)

    output_name = f"{language}-{stage}-pipeline"
    output_file = os.path.join("predictions", f"{output_name}.jsonl")
    os.makedirs("predictions", exist_ok=True)

    # Cache scanned repos to avoid re-scanning for datapoints from the same repo
    repo_cache: dict[str, dict[str, str]] = {}

    with jsonlines.open(input_file, "r") as reader:
        with jsonlines.open(output_file, "w") as writer:
            for datapoint in tqdm(reader, total=total, desc="Processing"):
                repo_path = datapoint["repo"].replace("/", "__")
                repo_revision = datapoint["revision"]
                root_dir = os.path.join(
                    "data",
                    f"repositories-{language}-{stage}",
                    f"{repo_path}-{repo_revision}",
                )

                cache_key = f"{repo_path}-{repo_revision}"
                if cache_key not in repo_cache:
                    repo_cache[cache_key] = scan_repo(root_dir, ext)

                repo_files = repo_cache[cache_key]
                active_path = datapoint["path"]

                result = process_datapoint(datapoint, repo_files, active_path, args)
                writer.write(result)

    print(f"\nPredictions written to: {output_file}")
    ctx_len = 0
    with jsonlines.open(output_file, "r") as reader:
        for entry in reader:
            ctx_len += len(entry.get("context", ""))
    print(f"Total context chars: {ctx_len:,}")
    print(f"Estimated tokens: ~{ctx_len / CHARS_PER_TOKEN:,.0f}")


if __name__ == "__main__":
    main()

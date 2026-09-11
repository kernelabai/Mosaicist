"""A tolerant PTX parser.

Parses just enough of PTX to fingerprint a kernel: module header, entry/func
headers with their performance directives, shared-memory declarations, and a
flat stream of labels and instructions per function body. Nested ``{ }``
scopes (common in CUTLASS inline asm) are flattened; vector operands such as
``{%f1, %f2}`` stay inside their instruction.

Line numbers refer to the original text, and ``.loc`` directives are attached
to the instructions that follow them so any instruction can be mapped back to
the source line (CuTeDSL with ``--generate-line-info``) that produced it.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

_IDENT = r"[A-Za-z_$%][\w$]*"
_LABEL_RE = re.compile(rf"^\s*({_IDENT})\s*:(?!:)")
_GUARD_RE = re.compile(r"^@(!?)(%?[\w$]+)\s+")
_DIM_RE = re.compile(r"\d+")

# Performance-tuning directives that may follow an entry's parameter list.
_PERF_DIRECTIVES = (
    "maxntid",
    "reqntid",
    "minnctapersm",
    "maxnctapersm",
    "maxnreg",
    "reqnctapercluster",
    "maxclusterrank",
    "explicitcluster",
    "noreturn",
    "pragma",
)

_TYPE_BYTES = {
    "b8": 1, "u8": 1, "s8": 1,
    "b16": 2, "u16": 2, "s16": 2, "f16": 2, "bf16": 2,
    "b32": 4, "u32": 4, "s32": 4, "f32": 4,
    "b64": 8, "u64": 8, "s64": 8, "f64": 8,
    "b128": 16,
}


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class Instr:
    index: int  # position in the function's instruction stream
    opcode: str  # full dotted opcode, e.g. "wgmma.wait_group.sync.aligned"
    operands: list[str]
    pred: str | None = None  # "%p1" or "!%p1"
    line: int = 0  # 1-based line in the PTX text
    loc: Loc | None = None  # source location from the last .loc directive

    @property
    def parts(self) -> list[str]:
        return self.opcode.split(".")

    @property
    def base(self) -> str:
        return self.parts[0]

    def text(self) -> str:
        guard = f"@{self.pred} " if self.pred else ""
        ops = ", ".join(self.operands)
        return f"{guard}{self.opcode} {ops};" if ops else f"{guard}{self.opcode};"


@dataclass
class Label:
    name: str
    line: int = 0


@dataclass
class SharedDecl:
    name: str
    nbytes: int | None  # None for `.extern .shared ... name[]` (dynamic smem)
    align: int = 1
    line: int = 0


@dataclass
class Function:
    name: str
    kind: str  # "entry" | "func"
    params: list[str] = field(default_factory=list)
    directives: dict[str, tuple[int, ...] | bool] = field(default_factory=dict)
    body: list[Instr | Label] = field(default_factory=list)
    shared: list[SharedDecl] = field(default_factory=list)
    line: int = 0

    @property
    def instrs(self) -> list[Instr]:
        return [s for s in self.body if isinstance(s, Instr)]


@dataclass
class Module:
    version: str | None = None
    target: str | None = None
    address_size: int | None = None
    functions: list[Function] = field(default_factory=list)
    shared: list[SharedDecl] = field(default_factory=list)
    files: dict[int, str] = field(default_factory=dict)

    @property
    def entries(self) -> list[Function]:
        return [f for f in self.functions if f.kind == "entry"]

    def entry(self, name: str | None = None) -> Function:
        """Return the named entry, the only entry, or the largest one."""
        entries = self.entries
        if name is not None:
            for e in entries:
                if e.name == name:
                    return e
            raise KeyError(f"no .entry named {name!r}; have {[e.name for e in entries]}")
        if not entries:
            raise ValueError("module has no .entry functions")
        return max(entries, key=lambda e: len(e.instrs))


class PTXParseError(ValueError):
    pass


def _strip_comments(text: str) -> str:
    """Blank out // and /* */ comments, preserving offsets and newlines."""
    out = list(text)
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def _split_labels(chunk: str) -> tuple[list[tuple[str, int]], str]:
    """Leading `name:` labels of a statement (with offsets) and the remainder."""
    labels, consumed, rest = [], 0, chunk
    while m := _LABEL_RE.match(rest):
        labels.append((m.group(1), consumed + m.start(1)))
        consumed += m.end()
        rest = rest[m.end() :]
    return labels, rest


# Directives terminated by a newline rather than ';'.
_LINE_DIRECTIVE_RE = re.compile(r"^([ \t]*\.(?:version|target|address_size|file|loc)\b[^;\n]*)$", re.M)


def split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside of (), [], {} nesting."""
    parts, depth, cur = [], 0, []
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching(text: str, open_pos: int) -> int:
    """Index of the '}' matching the '{' at open_pos."""
    depth = 0
    for m in re.finditer(r"[{}]", text[open_pos:]):
        depth += 1 if m.group() == "{" else -1
        if depth == 0:
            return open_pos + m.start()
    raise PTXParseError(f"unbalanced '{{' at offset {open_pos}")


def _shared_decl(stmt: str, line: int) -> SharedDecl | None:
    if ".shared" not in stmt.split():
        return None
    m = re.search(r"\.(b\d+|u\d+|s\d+|f\d+|bf16)\s+(" + _IDENT + r")\s*\[\s*(\d*)\s*\]", stmt)
    if not m:
        return None
    align_m = re.search(r"\.align\s+(\d+)", stmt)
    count = m.group(3)
    nbytes = int(count) * _TYPE_BYTES.get(m.group(1), 1) if count else None
    return SharedDecl(m.group(2), nbytes, int(align_m.group(1)) if align_m else 1, line)


def _parse_header(header: str, line: int) -> Function:
    kind = "entry" if re.search(r"\.entry\b", header) else "func"
    after_kw = re.split(r"\.(?:entry|func)\b", header, maxsplit=1)[1]
    # .func may have a return-parameter list before the name: .func (.param .b32 r) name(...)
    after_kw = after_kw.strip()
    if after_kw.startswith("("):
        close = after_kw.index(")")
        after_kw = after_kw[close + 1 :].strip()
    m = re.match(rf"({_IDENT})", after_kw)
    if not m:
        raise PTXParseError(f"cannot find function name in header at line {line}")
    fn = Function(name=m.group(1), kind=kind, line=line)
    rest = after_kw[m.end() :]
    if rest.lstrip().startswith("("):
        open_idx = rest.index("(")
        depth, close_idx = 0, None
        for i in range(open_idx, len(rest)):
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        if close_idx is None:
            raise PTXParseError(f"unterminated parameter list at line {line}")
        fn.params = split_top_level(rest[open_idx + 1 : close_idx])
        rest = rest[close_idx + 1 :]
    for dm in re.finditer(r"\.(\w+)([^.]*)", rest):
        name, args = dm.group(1), dm.group(2)
        if name not in _PERF_DIRECTIVES:
            continue
        nums = tuple(int(x) for x in _DIM_RE.findall(args))
        fn.directives[name] = nums if nums else True
    return fn


class _BodyParser:
    """Flattens nested `{ }` scopes into one stream, keeping labels scope-correct.

    PTX labels are scoped to their block, and CUTLASS inline asm reuses local
    names (`LAB_WAIT:` / `DONE:`) in every mbarrier-wait scope. Labels defined
    inside a nested scope are renamed `name@<scope>`, and each branch target is
    resolved against the innermost enclosing scope that defines it.
    """

    def __init__(self, fn: Function, files: dict[int, str], line_of):
        self.fn = fn
        self.files = files
        self.line_of = line_of
        self.loc: Loc | None = None
        self.n_instrs = 0
        self.scopes = [0]  # stack of open scope ids; 0 is the function body
        self.next_scope = 1
        self.scope_labels: dict[int, set[str]] = {0: set()}
        self.branches: list[tuple[Instr, tuple[int, ...]]] = []

    def feed(self, body: str, base: int) -> None:
        """Parse the text between an entry's outer braces; `base` is its offset."""
        start = 0
        for m in re.finditer(r"[;{}]", body):
            c, pos = m.group(), m.start()
            chunk = body[start:pos]
            if c == ";":
                self._statement(chunk, base + start)
                start = pos + 1
                continue
            # A brace at statement start opens/closes a scope; one after an
            # opcode belongs to a vector operand like {%f1, %f2}.
            labels, rest = _split_labels(chunk)
            if rest.strip():
                continue
            self._emit_labels(labels, base + start)
            if c == "{":
                self.scopes.append(self.next_scope)
                self.scope_labels[self.next_scope] = set()
                self.next_scope += 1
            elif len(self.scopes) > 1:
                self.scopes.pop()
            start = pos + 1
        labels, rest = _split_labels(body[start:])
        if rest.strip():
            raise PTXParseError(f"unterminated statement near line {self.line_of(base + start)}")
        self._emit_labels(labels, base + start)
        self._resolve_branches()

    def _emit_labels(self, labels: list[tuple[str, int]], offset: int) -> None:
        scope = self.scopes[-1]
        for name, rel in labels:
            self.scope_labels[scope].add(name)
            qualified = name if scope == 0 else f"{name}@{scope}"
            self.fn.body.append(Label(qualified, self.line_of(offset + rel)))

    def _resolve_branches(self) -> None:
        for ins, stack in self.branches:
            target = ins.operands[-1]
            for scope in reversed(stack):
                if target in self.scope_labels.get(scope, ()):
                    if scope != 0:
                        ins.operands[-1] = f"{target}@{scope}"
                    break

    def _statement(self, chunk: str, offset: int) -> None:
        # Labels may prefix the statement; account for their width in line numbers.
        labels, lstripped = _split_labels(chunk)
        self._emit_labels(labels, offset)
        lead = len(chunk) - len(lstripped)
        stmt = lstripped.strip()
        if not stmt:
            return
        line = self.line_of(offset + lead + (len(lstripped) - len(lstripped.lstrip())))
        if stmt.startswith("."):
            self._directive(stmt, line)
            return
        pred = None
        gm = _GUARD_RE.match(stmt)
        if gm:
            pred = gm.group(1) + gm.group(2)
            stmt = stmt[gm.end() :]
        pieces = stmt.split(None, 1)
        opcode = pieces[0]
        operands = split_top_level(pieces[1]) if len(pieces) > 1 else []
        ins = Instr(self.n_instrs, opcode, operands, pred, line, self.loc)
        self.n_instrs += 1
        self.fn.body.append(ins)
        if opcode.split(".", 1)[0] == "bra" and operands:
            self.branches.append((ins, tuple(self.scopes)))

    def _directive(self, stmt: str, line: int) -> None:
        word = stmt.split(None, 1)[0]
        if word == ".loc":
            nums = _DIM_RE.findall(stmt.split(",")[0])
            if len(nums) >= 2:
                fid, ln = int(nums[0]), int(nums[1])
                col = int(nums[2]) if len(nums) > 2 else 0
                self.loc = Loc(self.files.get(fid, f"file{fid}"), ln, col)
            return
        decl = _shared_decl(stmt, line)
        if decl is not None:
            self.fn.shared.append(decl)


def parse(text: str) -> Module:
    """Parse PTX text into a Module."""
    # Terminate newline-delimited directives with ';' so one scanner handles all
    # statements. Appending before the newline keeps line numbers intact.
    clean = _LINE_DIRECTIVE_RE.sub(r"\1;", _strip_comments(text))
    newlines = [i for i, c in enumerate(clean) if c == "\n"]

    def line_of(offset: int) -> int:
        return bisect.bisect_right(newlines, offset - 1) + 1

    mod = Module()
    pos, n = 0, len(clean)
    stmt_end = re.compile(r"[;{]")
    while pos < n:
        m = stmt_end.search(clean, pos)
        if not m:
            if clean[pos:].strip():
                raise PTXParseError(f"trailing text at line {line_of(pos)}")
            break
        header = clean[pos : m.start()]
        hstrip = header.strip()
        hline = line_of(pos + len(header) - len(header.lstrip()))
        if m.group() == ";":
            _module_statement(mod, hstrip, hline)
            pos = m.end()
            continue
        close = _find_matching(clean, m.start())
        if hstrip.endswith("="):  # array initializer: `.global .b8 x[4] = {1, 2, 3, 4};`
            semi = clean.index(";", close)
            pos = semi + 1
            continue
        if re.search(r"\.(?:entry|func)\b", hstrip):
            fn = _parse_header(hstrip, hline)
            _BodyParser(fn, mod.files, line_of).feed(clean[m.start() + 1 : close], m.start() + 1)
            mod.functions.append(fn)
        # anything else with a brace block (.section debug info, etc.) is skipped
        pos = close + 1
    return mod


def _module_statement(mod: Module, stmt: str, line: int) -> None:
    if not stmt:
        return
    words = stmt.split()
    head = words[0]
    if head == ".version" and len(words) > 1:
        mod.version = words[1]
    elif head == ".target" and len(words) > 1:
        mod.target = words[1].rstrip(",")
    elif head == ".address_size" and len(words) > 1:
        mod.address_size = int(words[1])
    elif head == ".file":
        m = re.match(r"\.file\s+(\d+)\s+\"([^\"]*)\"", stmt)
        if m:
            mod.files[int(m.group(1))] = m.group(2)
    elif re.search(r"\.(?:entry|func)\b", stmt):
        pass  # forward declaration
    else:
        decl = _shared_decl(stmt, line)
        if decl is not None:
            mod.shared.append(decl)

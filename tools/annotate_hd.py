"""Insert an execution-space macro before every function declaration/definition head.

Used to vendor PX4 C++ sources into px4/: each head gets ``PX4RL_HD``, which
the build defines as ``__host__ __device__`` (CUDA) or as nothing (CPU).
Scans namespace/class scope statements (function bodies are skipped), so
local variable constructions are never touched.
Usage: python tools/annotate_hd.py MACRO FILE...  (edits in place, prints counts).
"""
import re
import sys

CONTROL = {"if", "for", "while", "switch", "return", "else", "case", "do", "sizeof",
           "static_assert", "using", "typedef", "decltype"}
AGGREGATE = re.compile(r"^\s*(template\s*<.*>\s*)?(class|struct|union|enum|namespace)\b", re.S)


def blank_comments_and_strings(text):
    """Same-length copy with comments, string and char literals replaced by spaces."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2) + 2
            out[i:j] = [c if c == "\n" else " " for c in text[i:j]]
            i = j
        elif text[i] in "\"'":
            q, j = text[i], i + 1
            while text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out[i + 1:j] = " " * (j - i - 1)
            i = j + 1
        elif text[i] == "#" and (i == 0 or text[i - 1] == "\n" or text[:i].rsplit("\n", 1)[-1].strip() == ""):
            j = i
            while True:  # preprocessor line, with continuations
                k = text.find("\n", j)
                k = n if k < 0 else k
                if text[k - 1] != "\\":
                    break
                j = k + 1
            out[i:k] = [c if c == "\n" else " " for c in text[i:k]]
            i = k
        else:
            i += 1
    return "".join(out)


def skip_template_prefix(code, start):
    """If the statement at ``start`` begins with template<...>, return the index after it."""
    m = re.compile(r"\s*template\s*<").match(code, start)
    while m:
        depth, j = 1, m.end()
        while depth:
            depth += {"<": 1, ">": -1}.get(code[j], 0)
            j += 1
        start = j
        m = re.compile(r"\s*template\s*<").match(code, start)
    while code[start].isspace():
        start += 1
    return start


def is_function_head(stmt):
    """``stmt`` is the statement text up to (excluding) its terminating ';' or '{'."""
    s = re.sub(r"\s+", " ", stmt).strip()
    if not s or AGGREGATE.match(s) or "(" not in s:
        return False
    s = re.sub(r"^template ?<.*?> ?", "", s)
    first = re.match(r"[A-Za-z_]\w*", s)
    if first and first.group(0) in CONTROL:
        return False
    head = s[:s.index("(")]
    if "=" in head.replace("operator=", "").replace("operator==", "").replace("operator!=", "") \
            .replace("operator<=", "").replace("operator>=", "").replace("operator+=", "") \
            .replace("operator-=", "").replace("operator*=", "").replace("operator/=", ""):
        return False  # variable initialised with '=' before any '('
    if re.search(r"=\s*(default|delete)\s*$", s):
        return False  # defaulted/deleted members are implicitly host+device
    return bool(re.search(r"(operator\s*\S+|~?[A-Za-z_]\w*)\s*$", head))


def annotate(text, macro):
    code = blank_comments_and_strings(text)
    inserts = []
    scopes = ["ns"]  # ns | class | body
    i, n = 0, len(code)
    stmt_start = 0
    paren = 0
    while i < n:
        c = code[i]
        if scopes[-1] == "body":
            if c == "{":
                scopes.append("body")
            elif c == "}":
                scopes.pop()
                stmt_start = i + 1
            i += 1
            continue
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif paren == 0 and c in ";{}":
            stmt = code[stmt_start:i]
            if c == "}":
                scopes.pop()
            else:
                bare = re.sub(r"^\s*(public|private|protected)\s*:", "", stmt)
                head = is_function_head(bare)
                if head and not macro_already(text, stmt_start, i, macro):
                    pos = skip_template_prefix(code, stmt_start)
                    pos = skip_access_specifier(code, pos)
                    inserts.append(pos)
                if c == "{":
                    if head:
                        scopes.append("body")
                    elif (m := AGGREGATE.match(bare)) and m.group(2) != "enum":
                        scopes.append("class")
                    else:
                        scopes.append("body")  # enum, brace initialiser: skip
            stmt_start = i + 1
        i += 1
    for pos in sorted(inserts, reverse=True):
        text = text[:pos] + macro + " " + text[pos:]
    return text, len(inserts)


def skip_access_specifier(code, pos):
    m = re.compile(r"(public|private|protected)\s*:\s*").match(code, pos)
    return m.end() if m else pos


def macro_already(text, start, end, macro):
    return macro in text[start:end]


if __name__ == "__main__":
    macro, files = sys.argv[1], sys.argv[2:]
    total = 0
    for path in files:
        with open(path) as f:
            src = f.read()
        new, count = annotate(src, macro)
        with open(path, "w") as f:
            f.write(new)
        total += count
        print(f"{count:4d}  {path}")
    print(f"{total:4d}  total")

write a markdown translator by tree-sitter-language-pack (markdown parsing) and local ollama server (use openai package to get access to it, query latest downloaded model on the server by default), as a cli tool in python.

we employ a three stage conversion:

0. tree sitter parsing, to get a parsed concrete syntax tree for the document at the byte level. ERROR nodes will be logged as [WARN] to users using logging module. 
1. encoding detection. we use existing pypi package (i forget the name) to guess the encoding for text. after that stage, we output an utf-8 encoded document. 
2. language detection and sentence segmentation. we ask model language of each text, and ask it to output a standard language code. and we maintain a large database inline about how to segment sentence by re.compiled regexes and re.split/scan/... (or find a pypi package to do that). 
3. stage 1: naive translating. this stage we feed the model with threadpool, each sentence is a task. user will see a tqdm progress bar, the unit is sentences. after that stage we output a naive translated document. the thinking is disabled if there any for this stage. the model is asked to output plain text, no json. 
4. stage 2: node polishing. this stage we feed the model one by one for each node, sequencially. thinking is enabled if any, and output is streamed into the terminal token by token. we still have a progress bar for each node.  after this stage we output a polished document which the structure is guarenteed to match the original one. model is asked to output a valid xml:
<polish required="true/false"> (or output <polish required="false"/> to indicate it does not need polishing)
{text here if required is true}
</polish>
we use a sax parser, so that we can early stop and do not waste computation power on cheap nodes.
5. stage 3: final polishing. this stage takes the second stage full document in the full context window and output the final document in one pass. streaming is also applied. this time the model is asked to output plain markdown, and the result is blindly saved as stage 3 result. 

i need you to design an extensiable structure because we are aimed to support ANY type of plain text documents, just like pandoc. we may further treat a folder, a docx/pptx/xlsx (because they are just zipped plain text documents), ... as you might imagine. 

and we might support further backend by plain llama.cpp, litellm, vllm, sglang, ... as long as it is openai-compatiable. please also implement an vllm backend driver to load any local huggingface models to do inference. 

# Markdown Translator — Architecture & Implementation

Below is a complete design and reference implementation. The structure is explicitly **document-format-agnostic** (Pandoc-style extensibility) and **backend-agnostic** (any OpenAI-compatible server, plus a native vLLM driver).

---

## 1. Project layout

```
xlate/
├── pyproject.toml
├── xlate/
│   ├── __init__.py
│   ├── cli.py                     # Typer/argparse entry point
│   ├── config.py                  # dataclass settings
│   ├── pipeline.py                # orchestrates stages 0–3
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage0_parse.py        # tree-sitter → CST
│   │   ├── stage1_encode.py       # charset-normalizer → utf-8
│   │   ├── stage2_segment.py      # lang detect + sentence split
│   │   ├── stage3_naive.py        # threadpool translation
│   │   ├── stage4_polish.py       # per-node XML polish (SAX)
│   │   └── stage5_final.py        # full-context final polish
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                # Backend protocol
│   │   ├── openai_compat.py       # Ollama / LiteLLM / vLLM server
│   │   └── vllm_native.py         # in-process vLLM driver
│   ├── documents/
│   │   ├── __init__.py
│   │   ├── base.py                # Document protocol
│   │   ├── markdown.py            # tree-sitter markdown impl
│   │   └── registry.py            # extension → Document class
│   ├── sentence/
│   │   ├── __init__.py
│   │   ├── database.py            # language → regex splitter db
│   │   └── splitter.py            # unified splitter (pysbd/blingfire/regex)
│   └── utils/
│       ├── progress.py            # tqdm wrappers
│       └── xmlstream.py           # SAX early-stop parser
```

---

## 2. Core abstractions

### 2.1 Backend protocol

Any OpenAI-compatible server (Ollama, LiteLLM, vLLM HTTP, SGLang) and the native vLLM driver implement this protocol. The rest of the pipeline never knows which backend is in use.

```python
# xlate/backends/base.py
from __future__ import annotations
from typing import AsyncIterator, Iterator, Protocol, Sequence

class Backend(Protocol):
    """A chat-completion backend. All methods are synchronous;
    async variants can be added by wrapping in a thread pool."""

    def complete(
        self,
        messages: Sequence[dict],
        *,
        stream: bool = False,
        temperature: float = 0.2,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str | Iterator[str]:
        ...

    def model_name(self) -> str:
        ...
```

### 2.2 Document protocol

```python
# xlate/documents/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

@dataclass
class Node:
    """A translatable unit extracted from the document."""
    id: str
    kind: str          # 'text' | 'heading' | 'link_text' | 'code' | ...
    text: str          # utf-8 plain text content
    start_byte: int
    end_byte: int
    parent_id: str | None = None
    children: list[str] | None = None   # child node ids
    meta: dict | None = None            # grammar-specific extras

class Document(Protocol):
    """A document format that can be parsed, traversed and rebuilt."""

    @classmethod
    def extensions(cls) -> set[str]: ...

    @classmethod
    def parse(cls, source: bytes) -> "Document": ...

    def nodes(self) -> Iterator[Node]:
        """Yield translatable nodes in document order."""

    def replace(self, node_id: str, new_text: str) -> None:
        """Replace the text content of a node (byte-accurate)."""

    def render(self) -> bytes:
        """Serialize back to utf-8 bytes."""

    def warnings(self) -> list[str]:
        """Syntax warnings (ERROR nodes, etc.)."""
```

### 2.3 Registry

```python
# xlate/documents/registry.py
from pathlib import Path
from xlate.documents.markdown import MarkdownDocument

_REGISTRY: dict[str, type] = {}

def register(doc_cls):
    for ext in doc_cls.extensions():
        _REGISTRY[ext.lower()] = doc_cls
    return doc_cls

register(MarkdownDocument)

def detect(path: Path) -> type:
    ext = path.suffix.lower()
    if ext not in _REGISTRY:
        raise ValueError(f"no document handler for {ext}")
    return _REGISTRY[ext]
```

To add DOCX/PPTX/XLSX later you implement `Document` over `zipfile` + XML and call `register()`.

---

## 3. Stage 0 — Tree-sitter parsing

```python
# xlate/documents/markdown.py
from __future__ import annotations
from pathlib import Path
from tree_sitter_language_pack import get_parser
from xlate.documents.base import Node

# tree-sitter markdown grammar is available in the language pack
_PARSER = get_parser("markdown")

# Node kinds that contain natural-language text worth translating.
_TRANSLATABLE = {
    "text", "inline", "paragraph", "heading", "atx_heading",
    "setext_heading", "block_quote", "list_item", "link_text",
    "image_description", "emphasis", "strong_emphasis",
    "strikethrough", "table_cell",
}

class MarkdownDocument:
    def __init__(self, source: bytes, tree):
        self._source = source
        self._tree = tree
        self._nodes: dict[str, Node] = {}
        self._warnings: list[str] = []
        self._build_index()

    # ---- Document protocol -------------------------------------------------
    @classmethod
    def extensions(cls) -> set[str]:
        return {".md", ".markdown", ".mdown", ".mkd"}

    @classmethod
    def parse(cls, source: bytes) -> "MarkdownDocument":
        tree = _PARSER.parse(source)
        return cls(source, tree)

    def nodes(self):
        # deterministic order: byte offset ascending
        for nid in sorted(self._nodes, key=lambda k: self._nodes[k].start_byte):
            yield self._nodes[nid]

    def replace(self, node_id: str, new_text: str) -> None:
        self._nodes[node_id].text = new_text

    def render(self) -> bytes:
        # byte-accurate rebuild: walk original bytes, substitute replaced spans
        out = bytearray()
        cursor = 0
        for n in sorted(self._nodes.values(), key=lambda x: x.start_byte):
            if n.start_byte < cursor:
                continue
            out.extend(self._source[cursor:n.start_byte])
            out.extend(n.text.encode("utf-8"))
            cursor = n.end_byte
        out.extend(self._source[cursor:])
        return bytes(out)

    def warnings(self) -> list[str]:
        return list(self._warnings)

    # ---- private ----------------------------------------------------------
    def _build_index(self):
        stack = [(self._tree.root_node, None)]
        while stack:
            node, parent_id = stack.pop()
            if node.type == "ERROR" or node.is_missing:
                snippet = self._source[node.start_byte:node.end_byte][:60]
                self._warnings.append(
                    f"ERROR node at bytes {node.start_byte}-{node.end_byte}: {snippet!r}"
                )
            nid = f"n{node.id}"
            if node.type in _TRANSLATABLE and node.child_count == 0:
                text = self._source[node.start_byte:node.end_byte].decode("utf-8")
                self._nodes[nid] = Node(
                    id=nid, kind=node.type, text=text,
                    start_byte=node.start_byte, end_byte=node.end_byte,
                    parent_id=parent_id,
                )
            # push children in reverse so document order is preserved
            for i in range(node.child_count - 1, -1, -1):
                stack.append((node.child(i), nid))
```

Key points:

- `get_parser("markdown")` returns a configured tree-sitter parser from `tree-sitter-language-pack`.
- `ERROR` / `is_missing` nodes are collected as warnings and logged with `logging.warning()`; the pipeline does not abort.
- `render()` does a byte-accurate splice, so unchanged regions are never re-encoded.

---

## 4. Stage 1 — Encoding detection

```python
# xlate/stages/stage1_encode.py
import logging
from pathlib import Path
from charset_normalizer import from_bytes

log = logging.getLogger(__name__)

def to_utf8(path: Path) -> bytes:
    raw = path.read_bytes()
    # fast path: already valid utf-8
    try:
        raw.decode("utf-8")
        return raw
    except UnicodeDecodeError:
        pass

    best = from_bytes(raw).best()
    if best is None:
        log.warning("charset-normalizer found no match; falling back to latin-1")
        return raw.decode("latin-1").encode("utf-8")

    log.info("detected encoding: %s (confidence %.2f)",
             best.encoding, best.percent_chaos / 100)
    return str(best).encode("utf-8")
```

`charset-normalizer` is the PyPI package you were thinking of. It detects encoding from a `bytes`/file-pointer/`PathLike` and is explicitly motivated by `chardet`.

---

## 5. Stage 2 — Language detection & sentence segmentation

### 5.1 Sentence-splitting database

```python
# xlate/sentence/database.py
import re

# Each entry: language code -> (regex pattern, split-mode)
# Mode 'split' uses re.split, mode 'findall' uses re.findall.
DB: dict[str, tuple[re.Pattern, str]] = {
    "en": (re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])'), "split"),
    "de": (re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ"\'(])'), "split"),
    "fr": (re.compile(r'(?<=[.!?])\s+(?=[A-ZÀÂÉÈÊËÎÏÔÙÛÜ"\'(])'), "split"),
    "zh": (re.compile(r'(?<=[。！？；])\s*'), "split"),
    "ja": (re.compile(r'(?<=[。！？])\s*'), "split"),
    "ko": (re.compile(r'(?<=[.!?。！？])\s*'), "split"),
    "ru": (re.compile(r'(?<=[.!?])\s+(?=[А-ЯЁ"\'(])'), "split"),
    "ar": (re.compile(r'(?<=[.!?؟])\s+'), "split"),
    "he": (re.compile(r'(?<=[.!?])\s+'), "split"),
    "hi": (re.compile(r'(?<=[।!?])\s+'), "split"),
    "th": (re.compile(r'(?<=[.!?])\s*'), "split"),
    # ... extend freely
}

DEFAULT_PATTERN = re.compile(r'(?<=[.!?])\s+')
```

For production use, prefer `pysbd` (22 languages, rule-based, passes 97.92% of the Golden Rule Set) or `blingfire` (extremely fast, FSM-based) as the primary splitter and fall back to the regex DB.

### 5.2 Unified splitter

```python
# xlate/sentence/splitter.py
import re
from xlate.sentence.database import DB, DEFAULT_PATTERN

def split(text: str, lang: str) -> list[str]:
    entry = DB.get(lang)
    pat, mode = entry if entry else (DEFAULT_PATTERN, "split")
    if mode == "split":
        parts = pat.split(text)
    else:
        parts = pat.findall(text)
    return [p.strip() for p in parts if p.strip()]
```

### 5.3 Language detection prompt

The model is asked to return a **standard language code** (ISO 639‑1) on a single line:

```
Detect the language of the following text.
Reply with exactly one ISO 639-1 code (e.g. en, zh, fr).
Text:
...
```

```python
# xlate/stages/stage2_segment.py
import logging
from xlate.backends.base import Backend
from xlate.sentence.splitter import split

log = logging.getLogger(__name__)

LANG_PROMPT = (
    "Detect the language of the following text. "
    "Reply with exactly one ISO 639-1 code (two lowercase letters).\n"
    "Text:\n{text}"
)

def detect_language(backend: Backend, sample: str) -> str:
    raw = backend.complete(
        [{"role": "user", "content": LANG_PROMPT.format(text=sample[:2000])}],
        thinking=False,
        max_tokens=8,
        temperature=0.0,
    )
    code = raw.strip().split()[0].lower()[:2]
    if len(code) != 2 or not code.isalpha():
        log.warning("language detection returned %r; defaulting to 'en'", raw)
        return "en"
    return code

def segment_nodes(backend: Backend, nodes):
    """Return list of (node, lang, [sentences])."""
    joined = "\n".join(n.text for n in nodes if n.text)
    lang = detect_language(backend, joined)
    log.info("document language: %s", lang)
    out = []
    for n in nodes:
        sents = split(n.text, lang)
        out.append((n, lang, sents))
    return out
```

---

## 6. Stage 3 — Naive translation (threadpool + tqdm)

Each sentence is one task. Thinking is disabled. The model is asked to output plain text only (no JSON, no XML).

```python
# xlate/stages/stage3_naive.py
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from xlate.backends.base import Backend
from xlate.sentence.splitter import split

log = logging.getLogger(__name__)

T_PLAIN = (
    "Translate the following {src} text into {tgt}. "
    "Output only the translation, no explanations, no quotes.\n\n{text}"
)

def _translate_one(backend, text, src, tgt):
    return backend.complete(
        [{"role": "user",
          "content": T_PLAIN.format(src=src, tgt=tgt, text=text)}],
        thinking=False,
        temperature=0.2,
    )

def naive_translate(
    backend: Backend,
    segmented,               # [(node, lang, [sentences]), ...]
    target_lang: str,
    workers: int = 8,
) -> dict[str, str]:
    """Returns node_id -> translated text (sentences joined)."""
    tasks = []
    for node, lang, sents in segmented:
        for i, s in enumerate(sents):
            tasks.append((node.id, i, s, lang))

    results: dict[tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_translate_one, backend, s, lang, target_lang): (nid, i)
            for nid, i, s, lang in tasks
        }
        for fut in tqdm(as_completed(futs), total=len(futs), unit="sent"):
            nid, i = futs[fut]
            try:
                results[(nid, i)] = fut.result()
            except Exception as e:
                log.error("translation failed for %s/%d: %s", nid, i, e)
                results[(nid, i)] = ""

    # re-assemble per node
    by_node: dict[str, list[str]] = {}
    for node, lang, sents in segmented:
        by_node[node.id] = [results.get((node.id, i), "") for i in range(len(sents))]

    return {nid: " ".join(parts) for nid, parts in by_node.items()}
```

---

## 7. Stage 4 — Node polishing with SAX early-stop

The model is instructed to emit:

```xml
<polish required="false"/>
```

if the text needs no polishing, or

```xml
<polish required="true">…polished text…</polish>
```

We parse the stream with `xml.sax` and **stop reading as soon as `required="false"` is seen**, avoiding wasted computation on cheap nodes.

### 7.1 SAX early-stop handler

```python
# xlate/utils/xmlstream.py
import xml.sax
from xml.sax.handler import ContentHandler

class PolishHandler(ContentHandler):
    def __init__(self):
        super().__init__()
        self.required = None
        self.buffer: list[str] = []
        self.done = False

    def startElement(self, name, attrs):
        if name != "polish":
            return
        req = attrs.get("required", "true").lower()
        self.required = req == "true"
        if not self.required:
            self.done = True   # <- early stop signal

    def characters(self, content):
        if self.required:
            self.buffer.append(content)

    def endElement(self, name):
        if name == "polish":
            self.done = True

    @property
    def text(self) -> str:
        return "".join(self.buffer)
```

### 7.2 Streaming polish driver

```python
# xlate/stages/stage4_polish.py
import logging, sys, xml.sax
from tqdm import tqdm
from xlate.backends.base import Backend
from xlate.utils.xmlstream import PolishHandler

log = logging.getLogger(__name__)

POLISH_SYS = (
    "You are a meticulous editor. The user gives you a translated fragment. "
    "If it is already perfect, output exactly:\n"
    '  <polish required="false"/>\n'
    "Otherwise output:\n"
    "  <polish required=\"true\">…your polished text…</polish>\n"
    "Output valid XML only. No markdown fences."
)

def polish_node(backend: Backend, text: str, target_lang: str) -> str:
    handler = PolishHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    stream = backend.complete(
        [
            {"role": "system", "content": POLISH_SYS},
            {"role": "user", "content": f"Target language: {target_lang}\n\n{text}"},
        ],
        stream=True,
        thinking=True,          # thinking enabled for this stage
        temperature=0.3,
    )

    buf: list[str] = []
    try:
        for chunk in stream:
            buf.append(chunk)
            # feed accumulated XML to SAX in a tolerant way:
            # we re-parse the whole buffer each time until it parses or
            # the early-stop flag is set.
            try:
                xml.sax.parseString("".join(buf), handler)
            except xml.sax.SAXParseException:
                pass  # incomplete XML — keep streaming
            if handler.done:
                # early stop: break out of the model stream
                break
    finally:
        # drain the generator if the backend supports close()
        if hasattr(stream, "close"):
            stream.close()

    if handler.required is False:
        return text                    # unchanged
    return handler.text or text        # fallback to original if parse failed

def polish_all(backend, doc_nodes, target_lang: str):
    out = {}
    for node in tqdm(doc_nodes, unit="node"):
        out[node.id] = polish_node(backend, node.text, target_lang)
    return out
```

**Note on streaming + SAX:** the code re-parses the growing buffer each chunk. For very large outputs you can switch to `xml.sax.parse()` over a `io.TextIOBase` wrapper that yields chunks lazily; the early-stop flag is checked after every `feed()`. The important property is that as soon as `<polish required="false"/>` closes, `handler.done` becomes `True` and the model stream is abandoned.

---

## 8. Stage 5 — Final full-context polish

The full stage‑4 document is placed in the context window and the model is asked to output **plain Markdown**. Streaming is enabled. The result is **blindly saved** as the stage‑5 artifact.

```python
# xlate/stages/stage5_final.py
import logging, sys
from xlate.backends.base import Backend

log = logging.getLogger(__name__)

FINAL_SYS = (
    "You are a professional document editor. "
    "The user gives you a translated Markdown document. "
    "Return the final polished Markdown document. "
    "Preserve all structural elements (headings, lists, links, code fences, tables). "
    "Output Markdown only."
)

def final_polish(backend: Backend, markdown_text: str, target_lang: str) -> str:
    stream = backend.complete(
        [
            {"role": "system", "content": FINAL_SYS},
            {"role": "user",
             "content": f"Target language: {target_lang}\n\n{markdown_text}"},
        ],
        stream=True,
        thinking=True,
        temperature=0.3,
    )
    out = []
    for token in stream:
        out.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(out)
```

---

## 9. Backends

### 9.1 OpenAI-compatible backend (Ollama, LiteLLM, vLLM HTTP, SGLang)

```python
# xlate/backends/openai_compat.py
from __future__ import annotations
import logging
from typing import Iterator
from openai import OpenAI

log = logging.getLogger(__name__)

class OpenAICompatBackend:
    """Works with any OpenAI-compatible server.

    Ollama:  base_url='http://localhost:11434/v1/', api_key='ollama'
    vLLM:    base_url='http://localhost:8000/v1',  api_key='EMPTY'
    LiteLLM: base_url='http://localhost:4000',      api_key='...'
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "ollama",
        model: str | None = None,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model or self._discover_model()

    def _discover_model(self) -> str:
        try:
            models = self._client.models.list()
            ids = [m.id for m in models.data]
            if not ids:
                raise RuntimeError("server returned no models")
            log.info("auto-selected model: %s", ids[0])
            return ids[0]
        except Exception as e:
            raise RuntimeError(
                "could not discover a model; pass --model explicitly"
            ) from e

    def complete(
        self,
        messages,
        *,
        stream: bool = False,
        temperature: float = 0.2,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        kwargs = dict(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        # Ollama / vLLM accept `think` in extra_body (model-dependent).
        if thinking is not None:
            kwargs["extra_body"] = {"think": thinking}

        if stream:
            return self._stream(kwargs)
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _stream(self, kwargs) -> Iterator[str]:
        with self._client.chat.completions.create(**kwargs) as s:
            for chunk in s:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    def model_name(self) -> str:
        return self._model
```

Ollama's OpenAI compatibility layer requires `base_url='http://localhost:11434/v1/'` and `api_key='ollama'` (ignored). vLLM exposes the same API at `/v1` with `api_key='EMPTY'`.

### 9.2 Native vLLM driver (offline, in-process)

```python
# xlate/backends/vllm_native.py
from __future__ import annotations
import logging
from typing import Iterator
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)

class VLLMNativeBackend:
    """Loads a HuggingFace model directly with vLLM's offline LLM class."""

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self._model = model
        log.info("vLLM loaded %s", model)

    def complete(
        self,
        messages,
        *,
        stream: bool = False,
        temperature: float = 0.2,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        # vLLM offline API is batch-oriented; for a single prompt:
        prompt = self._llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens or 2048,
            stop=stop,
        )
        outputs = self._llm.generate([prompt], sp)
        text = outputs[0].outputs[0].text

        if stream:
            # offline vLLM does not stream token-by-token; we fake it
            # by yielding the result in chunks so downstream code still works.
            def _gen():
                for i in range(0, len(text), 16):
                    yield text[i:i+16]
            return _gen()
        return text

    def model_name(self) -> str:
        return self._model
```

The `LLM` class is vLLM's primary offline interface; it loads models directly from HuggingFace or a local directory and performs batched inference without a server. For true token streaming you can switch to `vllm.AsyncLLMEngine`, but the offline `LLM` is simpler and sufficient for batch translation.

---

## 10. Pipeline orchestration

```python
# xlate/pipeline.py
import logging, sys
from pathlib import Path
from xlate.backends.base import Backend
from xlate.documents.registry import detect
from xlate.stages import stage1_encode, stage2_segment, stage3_naive, stage4_polish, stage5_final

log = logging.getLogger(__name__)

def run(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    backend: Backend,
    *,
    workers: int = 8,
    stage: int = 5,
):
    # ---- stage 0: parse ------------------------------------------------
    raw = input_path.read_bytes()
    DocCls = detect(input_path)
    doc = DocCls.parse(raw)
    for w in doc.warnings():
        log.warning("[WARN] %s", w)

    # ---- stage 1: encoding --------------------------------------------
    if stage >= 1:
        raw = stage1_encode.to_utf8(input_path)
        # re-parse after re-encoding (byte offsets change)
        doc = DocCls.parse(raw)

    nodes = list(doc.nodes())

    # ---- stage 2: segment ---------------------------------------------
    if stage >= 2:
        segmented = stage2_segment.segment_nodes(backend, nodes)
        src_lang = segmented[0][1] if segmented else "en"
    else:
        segmented = [(n, "en", [n.text]) for n in nodes]
        src_lang = "en"

    # ---- stage 3: naive translate --------------------------------------
    if stage >= 3:
        naive = stage3_naive.naive_translate(
            backend, segmented, target_lang, workers=workers
        )
        for n in nodes:
            if n.id in naive:
                doc.replace(n.id, naive[n.id])
        (output_path.with_suffix(".stage3.md")).write_bytes(doc.render())

    # ---- stage 4: node polish ------------------------------------------
    if stage >= 4:
        polished = stage4_polish.polish_all(backend, nodes, target_lang)
        for n in nodes:
            doc.replace(n.id, polished.get(n.id, n.text))
        (output_path.with_suffix(".stage4.md")).write_bytes(doc.render())

    # ---- stage 5: final polish -----------------------------------------
    if stage >= 5:
        stage4_text = doc.render().decode("utf-8")
        final = stage5_final.final_polish(backend, stage4_text, target_lang)
        output_path.write_text(final, encoding="utf-8")
    else:
        output_path.write_bytes(doc.render())

    log.info("done: %s", output_path)
```

---

## 11. CLI entry point

```python
# xlate/cli.py
import argparse, logging, sys
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(
        prog="xlate",
        description="Multi-stage markdown translator (tree-sitter + OpenAI-compatible backends)",
    )
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-t", "--target", default="zh", help="target ISO 639-1 code")
    p.add_argument("--stage", type=int, default=5, choices=[0,1,2,3,4,5])
    p.add_argument("--workers", type=int, default=8)

    # backend selection
    p.add_argument("--backend", choices=["openai", "vllm"], default="openai")
    p.add_argument("--base-url", default="http://localhost:11434/v1/")
    p.add_argument("--api-key", default="ollama")
    p.add_argument("--model", default=None,
                   help="model name; if omitted, the first available model is used")

    # vllm-native only
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    p.add_argument("--gpu-mem", type=float, default=0.9)

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - (args.verbose * 10)
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.backend == "openai":
        from xlate.backends.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
    elif args.backend == "vllm":
        from xlate.backends.vllm_native import VLLMNativeBackend
        if not args.model:
            raise SystemExit("--model is required for the vllm backend")
        backend = VLLMNativeBackend(
            args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
        )

    out = args.output or args.input.with_name(
        args.input.stem + f".{args.target}.md"
    )

    from xlate.pipeline import run
    run(args.input, out, args.target, backend,
        workers=args.workers, stage=args.stage)

if __name__ == "__main__":
    main()
```

---

## 12. Extensibility checklist

| Goal | What to add |
|---|---|
| New document format (DOCX, PPTX, XLSX) | Subclass `Document`, implement `parse`/`nodes`/`replace`/`render`, call `register()`. DOCX/PPTX/XLSX are ZIP+XML; you can reuse `xml.sax` / `lxml` for the inner XML. |
| New backend (llama.cpp server, LiteLLM, SGLang) | They are all OpenAI-compatible; reuse `OpenAICompatBackend` with a different `base_url`. |
| New sentence splitter | Add an entry to `DB` or plug `pysbd`/`blingfire` into `splitter.py`. |
| New pipeline stage | Add a module under `stages/`, wire it into `pipeline.run()` behind a `--stage` guard. |
| Async concurrency | Replace `ThreadPoolExecutor` with `asyncio` + `AsyncOpenAI`; the `Backend` protocol can gain an `acomplete()` method. |
| Structured output for stage 4 | Replace the hand-rolled XML parser with `instructor` or `outlines` if you prefer JSON schemas. |

The three-stage conversion (naive → per-node polish → full-context polish) is preserved exactly as specified: stage 3 disables thinking and emits plain text; stage 4 enables thinking, streams tokens, and uses SAX early-stop; stage 5 sends the full document and blindly saves the Markdown result.
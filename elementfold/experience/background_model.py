# ElementFold · experience/background_model.py
# ──────────────────────────────────────────────────────────────────────────────
# BackgroundModel — tiny, portable LLM hand‑off (local ⇄ remote) with fallback
#
# Why this exists (plain words)
#   • We want a *local* small LLM available at all times (mandatory), and an
#     *optional* bigger LLM reachable over the LAN (second Jetson Orin).
#   • A single, tiny API manages both, checks health, and falls back cleanly.
#   • No hard deps at import time: heavy libraries are lazy‑imported inside
#     backends so ElementFold stays small when LLMs are not used.
#
# What you get
#   • BackgroundModel.generate(messages=...)  → Completion(text, meta)
#   • Health pings for both backends, with a tiny heartbeat thread.
#   • Backends included:
#       - LocalTransformers (if `transformers` is installed)
#       - LocalLlamaCpp      (if `llama_cpp` is installed; for .gguf)
#       - RemoteOpenAICompat (OpenAI/vLLM‑style /v1/chat/completions)
#       - RemoteOllama       (Ollama /api/chat)
#
# Design constraints
#   • Stdlib only at top‑level; backends import optional libs lazily.
#   • Friendly errors and explicit metadata so Studio/Interpreter can narrate.
#   • Thread‑safe health snapshots; simple lock, no complex concurrency.
#
# MIT‑style tiny utility. © 2025 ElementFold authors.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
import time
import threading
import traceback

# Type alias for chat messages: [{"role":"system"|"user"|"assistant","content":str}, ...]
Message = Dict[str, str]
Messages = List[Message]



# ──────────────────────────────────────────────────────────────────────────────
# Small data records
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PilotConfig:
    # — Local (mandatory) —
    local_kind: Literal["transformers", "llama_cpp", "ollama"] = "transformers"
    local_model: str = "tinyllama/TinyLlama-1.1B-Chat-v1.0"  # HF id | path | ollama tag
    local_device: Optional[str] = None       # "cuda:0" | "cpu" | None→auto
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0

    # transformers quantization (optional; ignored if bitsandbytes missing)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: Optional[str] = None  # "float16"|"bfloat16"|None

    # llama.cpp offload
    llama_n_ctx: int = 4096
    llama_n_gpu_layers: int = -1      # -1 = try to offload all layers to GPU
    llama_n_threads: Optional[int] = None

    # Local Ollama daemon (treated as "local" when chosen)
    local_ollama_url: str = "http://localhost:11434"

    # — Remote (optional) —
    remote_kind: Optional[Literal["openai", "ollama"]] = None
    remote_url: Optional[str] = None         # e.g., "http://jetson-b:8000" (no trailing slash)
    remote_model: Optional[str] = None
    remote_api_key: Optional[str] = None     # for OpenAI‑compat endpoints
    remote_timeout_s: float = 60.0

    # — Selection & health —
    prefer_remote: bool = False              # if remote is healthy, use it first
    fallback_on_error: bool = True
    heartbeat_every_s: float = 10.0
    request_timeout_s: float = 60.0
    verbose: bool = True


@dataclass
class Health:
    name: str
    kind: str
    healthy: bool
    last_error: Optional[str] = None
    last_latency_ms: Optional[float] = None
    last_checked_s: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Completion:
    text: str
    backend: str
    latency_ms: float
    finish_reason: str = "stop"
    tokens_new: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()

def _join_chat(messages: Messages) -> str:
    """
    Very small fallback chat template (when tokenizer.apply_chat_template is not available).
    """
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

def _safe_print(*a, **k) -> None:
    try:
        print(*a, **k)
    except Exception:
        pass

def _strip_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


# ──────────────────────────────────────────────────────────────────────────────
# Backend interface
# ──────────────────────────────────────────────────────────────────────────────

class _Backend:
    name: str
    kind: str
    def generate(self, messages: Messages, *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, timeout_s: float) -> Completion:
        raise NotImplementedError
    def health(self, timeout_s: float = 5.0) -> Health:
        raise NotImplementedError
    def close(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Local backends
# ──────────────────────────────────────────────────────────────────────────────

class _LocalTransformers(_Backend):
    """
    HF Transformers backend. Lazy‑loads tokenizer/model when first used.
    Works with chat templates if the tokenizer provides them.
    """
    def __init__(self, model_id: str, device: Optional[str], verbose: bool, cfg: PilotConfig):
        self.name = model_id
        self.kind = "transformers"
        self._device = device
        self._tok = None
        self._mdl = None
        self._verbose = verbose
        self._cfg = cfg

    def _ensure_loaded(self):
        if self._tok is not None and self._mdl is not None:
            return
        import torch  # local import
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("transformers is not installed; pip install transformers") from e

        if self._verbose:
            _safe_print(f"[pilot] loading transformers model: {self.name}")

        self._tok = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        # Make sure pad token exists to avoid warnings in generate()
        try:
            if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
                self._tok.pad_token = self._tok.eos_token  # type: ignore[attr-defined]
        except Exception:
            pass

        # Device / dtype / optional quantization
        kwargs: Dict[str, Any] = {}
        try:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        except Exception:
            pass

        # Optional bitsandbytes quantization
        if self._cfg.load_in_4bit or self._cfg.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                compute_dtype = None
                if self._cfg.bnb_4bit_compute_dtype:
                    # e.g., "float16" → torch.float16
                    import torch as _torch
                    compute_dtype = getattr(_torch, self._cfg.bnb_4bit_compute_dtype, None)
                quant = BitsAndBytesConfig(
                    load_in_4bit=bool(self._cfg.load_in_4bit),
                    load_in_8bit=bool(self._cfg.load_in_8bit),
                    bnb_4bit_quant_type=self._cfg.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                kwargs["quantization_config"] = quant
                kwargs.pop("torch_dtype", None)  # dtype ignored when quantized
            except Exception:
                # Silently ignore if bitsandbytes or config not available
                pass

        # Prefer device_map="auto" when device not pinned
        device_map = "auto" if self._device is None else None
        self._mdl = AutoModelForCausalLM.from_pretrained(self.name, device_map=device_map, **kwargs)
        if self._device is not None:
            self._mdl.to(self._device)
        self._mdl.eval()

    def _encode(self, messages: Messages):
        # Try chat template first (newer tokenizers)
        if hasattr(self._tok, "apply_chat_template"):
            try:
                ids = self._tok.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
                return ids
            except Exception:
                pass
        # Fallback prompt join
        text = _join_chat(messages)
        return self._tok(text, return_tensors="pt").input_ids

    def generate(self, messages: Messages, *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, timeout_s: float) -> Completion:
        import torch
        self._ensure_loaded()
        t0 = _now()
        ids = self._encode(messages)
        ids = ids.to(next(self._mdl.parameters()).device)

        do_sample = bool(temperature > 0.0)
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            eos_token_id=getattr(self._tok, "eos_token_id", None),
            pad_token_id=getattr(self._tok, "pad_token_id", None),
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(1e-8, float(temperature))
            if 0.0 < float(top_p) < 1.0:
                gen_kwargs["top_p"] = float(top_p)
            if int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

        with torch.no_grad():
            out = self._mdl.generate(ids, **gen_kwargs)

        gen = out[0, ids.shape[1]:]
        text = self._tok.decode(gen, skip_special_tokens=True)
        dt = (_now() - t0) * 1e3
        return Completion(text=text, backend=self.kind, latency_ms=dt,
                          finish_reason="stop", tokens_new=int(gen.numel()),
                          meta={"model": self.name})

    def health(self, timeout_s: float = 5.0) -> Health:
        try:
            t0 = _now()
            self._ensure_loaded()
            dt = (_now() - t0) * 1e3
            return Health(name=self.name, kind=self.kind, healthy=True,
                          last_checked_s=_now(), last_latency_ms=dt)
        except Exception as e:
            return Health(name=self.name, kind=self.kind, healthy=False,
                          last_error=str(e), last_checked_s=_now())


class _LocalLlamaCpp(_Backend):
    """
    llama.cpp backend via python‑llama‑cpp. Good for .gguf models (quantized).
    """
    def __init__(self, path_or_tag: str, verbose: bool, cfg: PilotConfig):
        self.name = path_or_tag
        self.kind = "llama_cpp"
        self._llm = None
        self._verbose = verbose
        self._cfg = cfg

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("llama_cpp is not installed; pip install llama-cpp-python") from e
        if self._verbose:
            _safe_print(f"[pilot] loading llama.cpp model: {self.name}")
        # Heuristic defaults; tweakable via PilotConfig
        self._llm = Llama(
            model_path=self.name,
            n_ctx=int(self._cfg.llama_n_ctx),
            n_threads=self._cfg.llama_n_threads,
            n_gpu_layers=int(self._cfg.llama_n_gpu_layers),
        )

    def generate(self, messages: Messages, *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, timeout_s: float) -> Completion:
        self._ensure_loaded()
        t0 = _now()
        # Prefer chat API if present
        try:
            res = self._llm.create_chat_completion(
                messages=messages,
                temperature=max(1e-8, float(temperature)),
                top_p=float(top_p),
                top_k=int(top_k) if top_k > 0 else None,
                max_tokens=int(max_new_tokens),
                stream=False,
            )
            text = res["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: join messages to a single prompt
            prompt = _join_chat(messages)
            res = self._llm(
                prompt=prompt,
                temperature=max(1e-8, float(temperature)),
                top_p=float(top_p),
                top_k=int(top_k) if top_k > 0 else None,
                max_tokens=int(max_new_tokens),
            )
            text = res["choices"][0]["text"]
        dt = (_now() - t0) * 1e3
        return Completion(text=text, backend=self.kind, latency_ms=dt,
                          meta={"model": self.name})

    def health(self, timeout_s: float = 5.0) -> Health:
        try:
            t0 = _now()
            self._ensure_loaded()
            dt = (_now() - t0) * 1e3
            return Health(name=self.name, kind=self.kind, healthy=True,
                          last_checked_s=_now(), last_latency_ms=dt)
        except Exception as e:
            return Health(name=self.name, kind=self.kind, healthy=False,
                          last_error=str(e), last_checked_s=_now())


# ──────────────────────────────────────────────────────────────────────────────
# Remote backends
# ──────────────────────────────────────────────────────────────────────────────

class _RemoteOpenAICompat(_Backend):
    """
    OpenAI/vLLM‑style endpoint:
      POST /v1/chat/completions
      payload: {"model": "...", "messages": [...], "temperature": ..., "top_p": ..., "max_tokens": ...}
      headers: {"Authorization":"Bearer <api_key>"} (optional)
    """
    def __init__(self, base_url: str, model: str, api_key: Optional[str], verbose: bool):
        self.kind = "openai"
        base = _strip_slash(base_url)
        self.name = f"{base}/v1/chat/completions"
        self._url = self.name
        self._model = model
        self._key = api_key
        self._verbose = verbose

    def _post_json(self, url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[int, str]:
        import urllib.request, urllib.error
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.getcode(), r.read().decode("utf-8")

    def generate(self, messages: Messages, *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, timeout_s: float) -> Completion:
        t0 = _now()
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": max(1e-8, float(temperature)),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
        }
        code, body = self._post_json(self._url, payload, timeout_s)
        if code // 100 != 2:
            raise RuntimeError(f"remote {self.kind} http {code}")
        obj = json.loads(body)
        text = obj["choices"][0]["message"]["content"]
        dt = (_now() - t0) * 1e3
        return Completion(text=text, backend=self.kind, latency_ms=dt,
                          finish_reason=obj["choices"][0].get("finish_reason", "stop"),
                          meta={"model": self._model})

    def health(self, timeout_s: float = 5.0) -> Health:
        # Fire a tiny test completion with max_tokens=1; tolerate errors with clear messages.
        try:
            _ = self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0, timeout_s=timeout_s
            )
            return Health(name=self.name, kind=self.kind, healthy=True,
                          last_checked_s=_now(), last_latency_ms=None)
        except Exception as e:
            return Health(name=self.name, kind=self.kind, healthy=False,
                          last_error=str(e), last_checked_s=_now())


class _RemoteOllama(_Backend):
    """
    Ollama chat endpoint:
      POST /api/chat
      payload: {"model":"<tag>", "messages":[...], "options":{"temperature":..., "num_predict":...}}
    """
    def __init__(self, base_url: str, model: str, verbose: bool):
        self.kind = "ollama"
        base = _strip_slash(base_url)
        self.name = f"{base}/api/chat"
        self._url = self.name
        self._model = model
        self._verbose = verbose

    def _post_json(self, url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[int, str]:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.getcode(), r.read().decode("utf-8")

    def generate(self, messages: Messages, *, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, timeout_s: float) -> Completion:
        t0 = _now()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": max(1e-8, float(temperature)),
                "top_p": float(top_p),
                "top_k": int(top_k) if top_k > 0 else None,
                "num_predict": int(max_new_tokens),
            }
        }
        code, body = self._post_json(self._url, payload, timeout_s)
        if code // 100 != 2:
            raise RuntimeError(f"remote {self.kind} http {code}")
        obj = json.loads(body)
        text = obj.get("message", {}).get("content", "")
        dt = (_now() - t0) * 1e3
        return Completion(text=text, backend=self.kind, latency_ms=dt,
                          meta={"model": self._model})

    def health(self, timeout_s: float = 5.0) -> Health:
        try:
            _ = self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0, timeout_s=timeout_s
            )
            return Health(name=self.name, kind=self.kind, healthy=True,
                          last_checked_s=_now(), last_latency_ms=None)
        except Exception as e:
            return Health(name=self.name, kind=self.kind, healthy=False,
                          last_error=str(e), last_checked_s=_now())


# ──────────────────────────────────────────────────────────────────────────────
# Front controller (hand‑off + heartbeat)
# ──────────────────────────────────────────────────────────────────────────────

class BackgroundModel:
    """
    Orchestrates a local and an optional remote LLM.
    Selection logic:
      • If prefer_remote and remote healthy → use remote.
      • Else use local (mandatory).
      • On any generation error, we optionally fall back to the other backend.
    """
    def __init__(self, cfg: Optional[PilotConfig] = None):
        self.cfg = cfg or PilotConfig()
        self._lock = threading.Lock()

        # Build local backend (mandatory)
        if self.cfg.local_kind == "transformers":
            self._local = _LocalTransformers(
                self.cfg.local_model, self.cfg.local_device, self.cfg.verbose, self.cfg
            )
        elif self.cfg.local_kind == "llama_cpp":
            self._local = _LocalLlamaCpp(self.cfg.local_model, self.cfg.verbose, self.cfg)
        elif self.cfg.local_kind == "ollama":
            # Local Ollama daemon is treated as "local" because it runs on the same machine.
            self._local = _RemoteOllama(self.cfg.local_ollama_url, self.cfg.local_model, self.cfg.verbose)
        else:
            raise ValueError(f"unknown local_kind: {self.cfg.local_kind}")

        # Build remote backend (optional)
        self._remote: Optional[_Backend] = None
        if self.cfg.remote_kind and self.cfg.remote_url and self.cfg.remote_model:
            if self.cfg.remote_kind == "openai":
                self._remote = _RemoteOpenAICompat(self.cfg.remote_url, self.cfg.remote_model,
                                                   self.cfg.remote_api_key, self.cfg.verbose)
            elif self.cfg.remote_kind == "ollama":
                self._remote = _RemoteOllama(self.cfg.remote_url, self.cfg.remote_model, self.cfg.verbose)
            else:
                raise ValueError(f"unknown remote_kind: {self.cfg.remote_kind}")

        # Health snapshots (start pessimistic; heartbeat will fill)
        self._h_local = Health(name=self._local.name, kind=self._local.kind, healthy=False)
        self._h_remote = Health(name=self._remote.name, kind=self._remote.kind, healthy=False) if self._remote else None

        # Heartbeat thread (daemon)
        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="PilotHeartbeat", daemon=True)
        self._hb_thread.start()

    # ——— public API ————————————————————————————————————————————————

    def close(self) -> None:
        self._hb_stop.set()
        try:
            if self._hb_thread.is_alive():
                self._hb_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._local.close()
        except Exception:
            pass
        try:
            if self._remote:
                self._remote.close()
        except Exception:
            pass

    def health(self) -> Dict[str, Any]:
        """Return a small JSON‑ready dict with backend health."""
        with self._lock:
            out = {"local": self._h_local.as_dict()}
            if self._h_remote:
                out["remote"] = self._h_remote.as_dict()
            return out

    def ready(self) -> bool:
        """True if at least the local backend is healthy."""
        with self._lock:
            return bool(self._h_local.healthy)

    def generate(self,
                 messages: Messages,
                 *,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 timeout_s: Optional[float] = None) -> Completion:
        """
        One‑shot chat completion with automatic backend selection and fallback.
        """
        max_new_tokens = int(max_new_tokens or self.cfg.max_new_tokens)
        temperature = float(temperature if temperature is not None else self.cfg.temperature)
        top_p = float(top_p if top_p is not None else self.cfg.top_p)
        top_k = int(top_k if top_k is not None else self.cfg.top_k)
        timeout_s = float(timeout_s or self.cfg.request_timeout_s)

        # Decide primary / secondary
        with self._lock:
            use_remote_first = bool(self.cfg.prefer_remote and self._remote and self._h_remote and self._h_remote.healthy)
        order: List[_Backend] = []
        if use_remote_first and self._remote:
            order = [self._remote, self._local]
        else:
            order = [self._local]
            if self._remote:
                order.append(self._remote)

        errors: List[str] = []
        for backend in order:
            try:
                out = backend.generate(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    timeout_s=timeout_s,
                )
                return out
            except Exception as e:
                msg = f"{backend.kind}:{backend.name} failed: {e}"
                errors.append(msg)
                if not self.cfg.fallback_on_error:
                    raise

        # If we reach here, all backends failed
        raise RuntimeError(" | ".join(errors) if errors else "no healthy LLM backend")

    # ——— internals ————————————————————————————————————————————————

    def _heartbeat_once(self) -> None:
        try:
            hloc = self._local.health(timeout_s=self.cfg.remote_timeout_s)
        except Exception as e:
            hloc = Health(name=self._local.name, kind=self._local.kind, healthy=False, last_error=str(e))
        with self._lock:
            self._h_local = hloc

        if self._remote:
            try:
                hrem = self._remote.health(timeout_s=self.cfg.remote_timeout_s)
            except Exception as e:
                hrem = Health(name=self._remote.name, kind=self._remote.kind, healthy=False, last_error=str(e))
            with self._lock:
                self._h_remote = hrem

    def _heartbeat_loop(self) -> None:
        # First ping quickly to prime health
        try:
            self._heartbeat_once()
        except Exception:
            pass
        # Then periodic pings
        while not self._hb_stop.wait(timeout=self.cfg.heartbeat_every_s):
            try:
                self._heartbeat_once()
            except Exception:
                # Keep the thread alive, but remember the error in snapshots
                err = traceback.format_exc(limit=1)
                with self._lock:
                    self._h_local.last_error = self._h_local.last_error or err
                    if self._h_remote:
                        self._h_remote.last_error = self._h_remote.last_error or err
    # ──────────────────────────────────────────────────────────────────────────────
    # Compatibility shim for LocalBrain
    # ──────────────────────────────────────────────────────────────────────────────

    def _chat(self, messages: Messages, **kw) -> Completion:
        """
        Wrapper so LocalBrain can call BackgroundModel.chat(messages=[...]).
        Simply forwards to .generate() and returns a Completion object.
        """
        return self.generate(messages, **kw)

    # attach only if missing
    if not hasattr(BackgroundModel, "chat"):
        BackgroundModel.chat = _chat  # type: ignore



# ──────────────────────────────────────────────────────────────────────────────
# Small convenience to build chat messages
# ──────────────────────────────────────────────────────────────────────────────

def as_messages(system: Optional[str] = None,
                user: Optional[str] = None,
                extra: Optional[Messages] = None) -> Messages:
    out: Messages = []
    if system:
        out.append({"role": "system", "content": system})
    if user:
        out.append({"role": "user", "content": user})
    if extra:
        out.extend(extra)
    return out

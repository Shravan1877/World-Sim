"""Exact, plain-data encoding of game objects for the LangGraph state and checkpoints (§11.6).

The graph state must be serializable. LangGraph's checkpointer (msgpack) can store dicts, lists,
str, int, float, bool, None and bytes without any custom types, so everything the graph keeps
(WorldState, the ledger, treaties, seat and turn records, pydantic decisions) is turned into those
plain types here, and back. The round trip is EXACT: numpy arrays keep their dtype, shape and bytes,
numpy scalars keep their dtype, tuples stay tuples, so a decoded state has the same state_hash()
and the same ledger (balances and transfer log) as the original.

Tags (a dict with one of these keys is an encoded object, never user data):
    "__nd__"  numpy array       {"__nd__": dtype, "shape": [...], "data": bytes}
    "__ng__"  numpy scalar      {"__ng__": dtype, "data": bytes}
    "__t__"   tuple             {"__t__": [...]}
    "__d__"   dict              {"__d__": [[key, value], ...]}   (keys may be tuples, accounts, ...)
    "__nt__"  NamedTuple        {"__nt__": class name, "v": [...]}
    "__dc__"  dataclass         {"__dc__": class name, "f": {field: value}}
    "__pm__"  pydantic model    {"__pm__": class name, "v": model_dump()}
    "__lg__"  Ledger            opening and balances (all_accounts order) + columnar transfer log
Only classes in the registry below can be decoded (no arbitrary imports).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from pydantic import BaseModel

from world import actions as actions_mod
from world import history as history_mod
from world.engine import shocks as shocks_mod
from world.engine import state as state_mod
from world.engine import step as step_mod
from world.engine import trade as trade_mod
from world.engine import treaties as treaties_mod
from world.engine.policy import PolicyEffects
from world.ledger import Account, Ledger, Transfer, all_accounts
from world.policies.base import DecisionResult


def _classes(*modules) -> dict[str, type]:
    out: dict[str, type] = {}
    for m in modules:
        for name, obj in vars(m).items():
            if isinstance(obj, type) and obj.__module__ == m.__name__:
                out[name] = obj
    return out


_REGISTRY: dict[str, type] = {
    **_classes(state_mod, history_mod, shocks_mod, treaties_mod, step_mod, trade_mod, actions_mod),
    "PolicyEffects": PolicyEffects,
    "DecisionResult": DecisionResult,
    "Account": Account,
    "Transfer": Transfer,
}
_ACCOUNTS = all_accounts()
_ACC_INDEX = {a: k for k, a in enumerate(_ACCOUNTS)}


def _cls(name: str) -> type:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"serial: class {name!r} is not in the registry") from None


def encode(obj: Any) -> Any:
    """Game object -> plain data (dict/list/str/int/float/bool/None/bytes)."""
    if obj is None or isinstance(obj, bool | str | bytes):
        return obj
    if isinstance(obj, np.generic):  # before int/float: np.float64 is a float subclass
        return {"__ng__": obj.dtype.str, "data": obj.tobytes()}
    if isinstance(obj, int | float):
        return obj
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__nd__": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(obj, Ledger):
        return _encode_ledger(obj)
    if isinstance(obj, BaseModel):
        return {"__pm__": type(obj).__name__, "v": encode(obj.model_dump())}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__dc__": type(obj).__name__,
            "f": {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return {"__nt__": type(obj).__name__, "v": [encode(v) for v in obj]}
    if isinstance(obj, tuple):
        return {"__t__": [encode(v) for v in obj]}
    if isinstance(obj, list):
        return [encode(v) for v in obj]
    if isinstance(obj, dict):
        return {"__d__": [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f"serial: cannot encode {type(obj).__name__}")


def decode(data: Any) -> Any:
    """Plain data -> game object (inverse of encode)."""
    if isinstance(data, list):
        return [decode(v) for v in data]
    if not isinstance(data, dict):
        return data
    if "__nd__" in data:
        return np.frombuffer(data["data"], dtype=np.dtype(data["__nd__"])).reshape(data["shape"]).copy()
    if "__ng__" in data:
        return np.frombuffer(data["data"], dtype=np.dtype(data["__ng__"]))[0]
    if "__t__" in data:
        return tuple(decode(v) for v in data["__t__"])
    if "__d__" in data:
        return {_hashable(decode(k)): decode(v) for k, v in data["__d__"]}
    if "__nt__" in data:
        return _cls(data["__nt__"])(*[decode(v) for v in data["v"]])
    if "__dc__" in data:
        return _cls(data["__dc__"])(**{k: decode(v) for k, v in data["f"].items()})
    if "__pm__" in data:
        return _cls(data["__pm__"]).model_validate(decode(data["v"]))
    if "__lg__" in data:
        return _decode_ledger(data)
    raise ValueError(f"serial: unknown encoded object with keys {sorted(data)}")


def _hashable(k: Any) -> Any:
    return tuple(k) if isinstance(k, list) else k


def _encode_ledger(led: Ledger) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    src = np.array([_ACC_INDEX[t.src] for t in led.log], dtype=np.int16)
    dst = np.array([_ACC_INDEX[t.dst] for t in led.log], dtype=np.int16)
    amt = np.array([t.amount for t in led.log], dtype=np.float64)
    why = np.array([reasons.setdefault(t.reason, len(reasons)) for t in led.log], dtype=np.int32)
    return {
        "__lg__": 1,
        "opening": encode(np.array([led.opening[a] for a in _ACCOUNTS], dtype=np.float64)),
        "balances": encode(np.array([led.balances[a] for a in _ACCOUNTS], dtype=np.float64)),
        "reasons": list(reasons),
        "src": encode(src),
        "dst": encode(dst),
        "amount": encode(amt),
        "reason": encode(why),
    }


def _decode_ledger(data: dict[str, Any]) -> Ledger:
    opening, balances = decode(data["opening"]), decode(data["balances"])
    led = Ledger({a: float(opening[k]) for k, a in enumerate(_ACCOUNTS)})
    led.balances = {a: float(balances[k]) for k, a in enumerate(_ACCOUNTS)}
    reasons = data["reasons"]
    src, dst, amt, why = (decode(data[k]) for k in ("src", "dst", "amount", "reason"))
    led.log = [
        Transfer(_ACCOUNTS[s], _ACCOUNTS[d], float(a), reasons[r])
        for s, d, a, r in zip(src.tolist(), dst.tolist(), amt.tolist(), why.tolist(), strict=True)
    ]
    return led

from __future__ import annotations


def dynamic_import(path: str):
    mod, _, cls = path.rpartition(".")
    module = __import__(mod, fromlist=[cls])
    return getattr(module, cls)

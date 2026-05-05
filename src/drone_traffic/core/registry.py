from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {
    "backbone": {},
    "detector": {},
    "tracker": {},
    "fusion": {},
    "cmc": {},
}


def register_backbone(name: str) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        _REGISTRY["backbone"][name] = cls
        return cls

    return wrapper


def register_detector(name: str) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        _REGISTRY["detector"][name] = cls
        return cls

    return wrapper


def register_tracker(name: str) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        _REGISTRY["tracker"][name] = cls
        return cls

    return wrapper


def register_fusion(name: str) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        _REGISTRY["fusion"][name] = cls
        return cls

    return wrapper


def register_cmc(name: str) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        _REGISTRY["cmc"][name] = cls
        return cls

    return wrapper


def get_registry(category: str) -> dict[str, type]:
    return _REGISTRY.get(category, {}).copy()


def build_component(category: str, name: str, **kwargs: Any) -> Any:
    components = _REGISTRY.get(category, {})
    if name not in components:
        available = list(components.keys())
        raise ValueError(
            f"Unknown {category} '{name}'. Available: {available}"
        )
    return components[name](**kwargs)

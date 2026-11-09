"""DCBF active-learning workflow package."""

__all__ = ["WorkspaceBootstrapper", "GenerationRunner", "DCBFReducer"]


def __getattr__(name):
    if name == "WorkspaceBootstrapper":
        from .bootstrap import WorkspaceBootstrapper

        return WorkspaceBootstrapper
    if name == "GenerationRunner":
        from .generation import GenerationRunner

        return GenerationRunner
    if name == "DCBFReducer":
        from .reduce import DCBFReducer

        return DCBFReducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

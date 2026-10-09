"""Read Compose service metadata without changing Docker's original input."""

import yaml


def _empty_reset(loader, node):
    if not isinstance(loader, yaml.SafeLoader):
        raise TypeError("Compose metadata requires SafeLoader")
    if not isinstance(node, yaml.SequenceNode) or node.value:
        raise ValueError(f"only empty-list !reset is supported at {node.start_mark}")
    return []


class _ComposeMetadataLoader(yaml.SafeLoader):
    pass


_ComposeMetadataLoader.add_constructor("!reset", _empty_reset)


def load_compose_metadata(text):
    """Accept empty-list reset tags for inspection; never rewrite Compose files."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Compose metadata must be nonempty YAML text")
    document = yaml.load(text, Loader=_ComposeMetadataLoader)
    if not isinstance(document, dict):
        raise ValueError("Compose document must be a mapping")
    return document

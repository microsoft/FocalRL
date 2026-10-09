"""Install the narrowly scoped Compose metadata reader in an existing Harbor tree."""

import argparse
import ast
from pathlib import Path

from adaptive_branching.src.swe import harbor_compose

OLD = "            document = yaml.safe_load(compose_path.read_text())"
NEW = (
    "            from agent_server.compose_metadata import load_compose_metadata\n"
    "            document = load_compose_metadata(compose_path.read_text())"
)


def patch_source(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Harbor Docker source must be nonempty")
    ast.parse(source)
    if NEW in source:
        if source.count(NEW) != 1 or OLD in source:
            raise ValueError("ambiguous existing Compose patch")
        return source
    if source.count(OLD) != 1:
        raise ValueError("expected exactly one Compose metadata loading site")
    result = source.replace(OLD, NEW, 1)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    docker = args.source / "src/harbor/environments/docker/docker.py"
    target = args.source / "agent_server/compose_metadata.py"
    if not docker.is_file() or not target.parent.is_dir():
        raise ValueError(f"not a Harbor source tree: {args.source}")
    original = docker.read_text()
    patched = patch_source(original)
    helper = Path(harbor_compose.__file__).read_text()
    backup = docker.with_name("docker.py.before-compose-reset")
    if not backup.exists():
        backup.write_text(original)
    elif patch_source(backup.read_text()) != patched:
        raise ValueError("backup does not match current source")
    temporary = target.with_suffix(".py.tmp")
    temporary.write_text(helper)
    temporary.replace(target)
    temporary = docker.with_suffix(".py.tmp")
    temporary.write_text(patched)
    temporary.replace(docker)
    print(f"PASS: Compose metadata patch installed in {args.source}")


if __name__ == "__main__":
    main()

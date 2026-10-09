import pytest
import yaml

from adaptive_branching.src.swe.harbor_compose import load_compose_metadata
from adaptive_branching.tools.swe.patch_harbor_compose import NEW, OLD, patch_source


@pytest.mark.parametrize("tag", ["", "!reset "])
def test_dns_reset_preserves_service_metadata(tag):
    text = f"services:\n  main:\n    dns: {tag}[]\n    extra_hosts: {tag}[]\n    image: example:tag\n"
    document = load_compose_metadata(text)
    assert document == {"services": {"main": {"dns": [], "extra_hosts": [], "image": "example:tag"}}}
    assert ("!reset" in text) == bool(tag)


def test_safe_loader_remains_unchanged():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("dns: !reset []")


@pytest.mark.parametrize("text", ["", " ", None, "[]", "null"])
def test_invalid_document(text):
    with pytest.raises(ValueError):
        load_compose_metadata(text)


@pytest.mark.parametrize("value", ["[server]", "{}", "null", "true"])
def test_unsupported_reset_fails(value):
    with pytest.raises(ValueError, match="only empty-list"):
        load_compose_metadata(f"dns: !reset {value}")


@pytest.mark.parametrize("tag", ["!unknown", "!!python/object/apply:os.system"])
def test_unknown_tags_remain_rejected(tag):
    with pytest.raises(yaml.constructor.ConstructorError):
        load_compose_metadata(f"dns: {tag} []")


def test_empty_mapping_and_explicit_networking():
    assert load_compose_metadata("{}") == {}
    text = "services:\n  main:\n    network_mode: host\n    dns: !reset []\n"
    assert load_compose_metadata(text)["services"]["main"]["network_mode"] == "host"


def test_patch_is_idempotent_and_only_changes_reader():
    source = "def inspect():\n    for compose_path in []:\n" + OLD + "\n"
    patched = patch_source(source)
    assert patched == source.replace(OLD, NEW)
    assert patch_source(patched) == patched


@pytest.mark.parametrize("source", ["", None, "pass", "def inspect():\n" + OLD + "\n" + OLD])
def test_patch_rejects_missing_or_duplicate_site(source):
    with pytest.raises(ValueError):
        patch_source(source)


def test_install_backup_and_repeat(tmp_path, monkeypatch):
    from adaptive_branching.tools.swe.patch_harbor_compose import main

    monkeypatch.setattr("sys.argv", ["patch", str(tmp_path)])
    with pytest.raises(ValueError, match="not a Harbor"):
        main()
    docker = tmp_path / "src/harbor/environments/docker/docker.py"
    docker.parent.mkdir(parents=True)
    (tmp_path / "agent_server").mkdir()
    original = "def inspect():\n    for compose_path in []:\n" + OLD + "\n"
    docker.write_text(original)
    main()
    assert docker.read_text() == patch_source(original)
    backup = docker.with_name("docker.py.before-compose-reset")
    assert backup.read_text() == original
    assert (tmp_path / "agent_server/compose_metadata.py").is_file()
    main()
    assert backup.read_text() == original
    backup.write_text(original + "\n# different source\n")
    with pytest.raises(ValueError, match="backup does not match"):
        main()

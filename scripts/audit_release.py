"""Check publishable text for credentials, private endpoints and host identity.

Findings contain file/line and rule names only; matched values are never printed.
Use --deny-file for additional private identifiers (keep that file outside the repo).
"""
from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
import subprocess

RULES = {
    "credential": re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|wandb_v1_[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    "signed-url": re.compile(r"[?&](?:sig|X-Amz-Signature)=[A-Za-z0-9%/+]{16,}", re.I),
    "home-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@(?:gmail|outlook|hotmail|protonmail|qq|163)\.[A-Za-z]+\b"),
}
IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
SKIP = {".git", "__pycache__", ".pytest_cache", ".cache", ".venv", "build", "dist"}


def scan_text(text: str, deny: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    if not isinstance(text, str) or any(not isinstance(word, str) or not word.strip() for word in deny):
        raise ValueError("text must be a string and deny identifiers must be nonempty strings")
    identifiers = [re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.I) for word in deny]
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in RULES.items():
            if pattern.search(line):
                findings.append((number, rule))
        for match in IP.finditer(line):
            try:
                ip = ipaddress.ip_address(match.group())
            except ValueError:
                continue  # not an IP (e.g. a package version)
            if (ip.is_private or ip in ipaddress.ip_network((1681915904, 10))) and not (ip.is_loopback or ip.is_unspecified):
                findings.append((number, "private-address"))
        if any(pattern.search(line) for pattern in identifiers):
            findings.append((number, "private-identifier"))
    return findings


def audit(root: Path, deny: tuple[str, ...] = (), *, tracked: bool = False) -> list[str]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    root = root.resolve()
    if tracked:
        names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
        paths = [root / name for name in names if name]
        if not paths:
            raise ValueError("tracked audit requires a nonempty Git index")
    else:
        paths = [p for p in root.rglob("*") if p.is_file() and not SKIP.intersection(p.relative_to(root).parts)]
    findings = []
    for path in sorted(paths):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"{relative}:0: symlink")
            continue
        if path.name.startswith(".env") or path.suffix in {".pem", ".key", ".p12", ".pt", ".sqlite", ".safetensors"}:
            findings.append(f"{relative}:0: private-artifact")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}:0: unreviewed-binary")
            continue
        findings.extend(f"{relative}:{line}: {rule}" for line, rule in scan_text(text, deny))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deny-file", type=Path)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    deny = ()
    if args.deny_file is not None:
        if not args.deny_file.is_file():
            raise FileNotFoundError(args.deny_file)
        if args.deny_file.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("private deny-file must live outside the published repository")
        deny = tuple(line.strip() for line in args.deny_file.read_text().splitlines() if line.strip())
    findings = audit(args.root, deny, tracked=args.tracked)
    if findings:
        print("\n".join(findings))
        raise SystemExit(1)
    print("PASS: no findings in the audited release files")


if __name__ == "__main__":
    main()

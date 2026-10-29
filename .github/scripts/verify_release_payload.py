#!/usr/bin/env python3
"""Fail closed if an SDK release archive contains non-public payload files."""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import PurePosixPath

PRIVATE_MARKERS = (
    re.compile(rb"VectorAmpCore", re.IGNORECASE),
    re.compile(rb"BlueAleph", re.IGNORECASE),
    re.compile(rb"(?:/builds/|/Users/ec2-user/|/home/hilbert/)", re.IGNORECASE),
    re.compile(rb"(?:research/sable|crates/vst-)", re.IGNORECASE),
    re.compile(rb"gitlab\.com/VectorAmp", re.IGNORECASE),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY", re.IGNORECASE),
    re.compile(rb"glpat-[A-Za-z0-9_-]{12,}"),
    re.compile(rb"npm_[A-Za-z0-9]{20,}"),
)
NATIVE_SUFFIXES = {".a", ".dll", ".dylib", ".node", ".o", ".rlib", ".so"}
SDIST_ROOT_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "LICENSE",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
}


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SystemExit(f"unsafe archive path: {name}")
    return path


def scan_payload(name: str, payload: bytes) -> None:
    path = safe_path(name)
    if path.suffix.lower() in NATIVE_SUFFIXES:
        raise SystemExit(f"SDK archive must not vendor a native runtime: {name}")
    for marker in PRIVATE_MARKERS:
        if marker.search(payload):
            raise SystemExit(f"private marker found in release payload: {name}")


def verify_wheel(archive: str) -> None:
    with zipfile.ZipFile(archive) as wheel:
        for info in wheel.infolist():
            path = safe_path(info.filename)
            if not (
                path.parts[0] == "vectoramp"
                or (len(path.parts) > 1 and path.parts[0].endswith(".dist-info"))
            ):
                raise SystemExit(f"unexpected wheel member: {info.filename}")
            scan_payload(info.filename, wheel.read(info))


def verify_sdist(archive: str) -> None:
    with tarfile.open(archive, "r:gz") as sdist:
        for member in sdist.getmembers():
            path = safe_path(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f"unsupported sdist member type: {member.name}")
            if member.isdir():
                continue
            relative = PurePosixPath(*path.parts[1:])
            if not (
                relative.as_posix() in SDIST_ROOT_FILES
                or relative.parts[:2] == ("src", "vectoramp")
            ):
                raise SystemExit(f"unexpected sdist member: {member.name}")
            extracted = sdist.extractfile(member)
            if extracted is None:
                raise SystemExit(f"could not read sdist member: {member.name}")
            scan_payload(member.name, extracted.read())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: verify_release_payload.py ARCHIVE...")
    for archive in sys.argv[1:]:
        if archive.endswith(".whl"):
            verify_wheel(archive)
        elif archive.endswith(".tar.gz"):
            verify_sdist(archive)
        else:
            raise SystemExit(f"unsupported release archive: {archive}")
    print(f"verified {len(sys.argv) - 1} public SDK release archives")


if __name__ == "__main__":
    main()

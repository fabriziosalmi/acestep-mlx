"""Model files the engine needs, fetched from Hugging Face on the first start.

Every source is pinned to a repository revision. After a source is downloaded, the sha256 of
each of its LFS files (the weights) is checked against the repository's metadata, then a
marker is written under ``<checkpoints>/.acestep-mlx/``; a source with a marker is taken as
present, so later starts need no network. Other files (configs, tokenizer files, model code)
are only required to exist: upstream rewrites the model code files in place.
"""

import fnmatch
import hashlib
import json
import os
import shutil
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

MARKER_DIR = ".acestep-mlx"
# Free space kept beyond the files themselves: bytecode, songs, the system's own needs.
SPARE_BYTES = 1_000_000_000


@dataclass(frozen=True)
class Source:
    """A pinned Hugging Face repository, or the part of it that ``patterns`` match."""

    name: str
    repo: str
    revision: str
    # fnmatch patterns on repository paths, as huggingface_hub's allow_patterns; None: all.
    patterns: Optional[Tuple[str, ...]] = None
    # Folder inside checkpoints/ that receives the repository's files; "" is checkpoints/.
    subdir: str = ""

    def wants(self, path: str) -> bool:
        """True if the repository file ``path`` belongs to this source."""
        return self.patterns is None or any(fnmatch.fnmatch(path, p) for p in self.patterns)


# DiT, VAE and text encoder from upstream's main repository (bench/f0/setup.sh downloads the same set).
MAIN = Source("main", "ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22",
              ("acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*"))
LMS = {
    "acestep-5Hz-lm-0.6B": Source("acestep-5Hz-lm-0.6B", "ACE-Step/acestep-5Hz-lm-0.6B",
                                  "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
                                  subdir="acestep-5Hz-lm-0.6B"),
}


@dataclass(frozen=True)
class RemoteFile:
    """One repository file: its path, size in bytes, and sha256 when it is an LFS file."""

    path: str
    size: int
    sha256: Optional[str] = None


class ModelError(RuntimeError):
    """The model files could not be fetched or did not match the repository."""


def sources_for(lm_model: str) -> List[Source]:
    """The sources a preset with ``lm_model`` needs.

    Raises:
        ModelError: If no download source is known for the LM.
    """
    if lm_model not in LMS:
        raise ModelError(f"no download source for LM {lm_model!r}")
    return [MAIN, LMS[lm_model]]


def remote_files(source: Source) -> List[RemoteFile]:
    """The files of ``source`` at its revision, from the Hugging Face API."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(source.repo, revision=source.revision, files_metadata=True)
    return [RemoteFile(s.rfilename, s.size or 0, s.lfs.sha256 if s.lfs else None)
            for s in info.siblings if source.wants(s.rfilename)]


def fetch(source: Source, target: str) -> None:
    """Download ``source`` into ``target``; files already complete there are kept."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=source.repo, revision=source.revision, local_dir=target,
                      allow_patterns=list(source.patterns) if source.patterns else None)


def marker_path(checkpoints: str, source: Source) -> str:
    """The file that records ``source`` as downloaded and checked."""
    return os.path.join(checkpoints, MARKER_DIR, f"{source.name}-{source.revision}.ok")


def target_dir(checkpoints: str, source: Source) -> str:
    """Where ``source``'s files land."""
    return os.path.join(checkpoints, source.subdir) if source.subdir else checkpoints


def bytes_present(target: str, files: Sequence[RemoteFile]) -> int:
    """Bytes of ``files`` already in ``target``, counting downloads in progress.

    A finished file appears at its path only once complete; partial downloads are the
    ``*.incomplete`` files huggingface_hub keeps under ``.cache/huggingface/download``.
    Those are created at their full size and filled in place, so a partial file counts by
    the blocks written to disk, not by its size.
    """
    done = 0
    for f in files:
        path = os.path.join(target, f.path)
        if os.path.isfile(path):
            done += min(os.path.getsize(path), f.size)
    partial_dir = os.path.join(target, ".cache", "huggingface", "download")
    for root, _dirs, names in os.walk(partial_dir):
        for name in names:
            if name.endswith(".incomplete"):
                st = os.stat(os.path.join(root, name))
                done += min(st.st_blocks * 512, st.st_size)
    return done


def sha256_of(path: str) -> str:
    """Hex sha256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(target: str, files: Sequence[RemoteFile]) -> None:
    """Every file exists in ``target``; every LFS file has its size and sha256.

    Raises:
        ModelError: For the first file that is missing or differs.
    """
    for f in files:
        path = os.path.join(target, f.path)
        if not os.path.isfile(path):
            raise ModelError(f"{f.path} is missing after the download")
        if f.sha256 is None:
            continue
        if os.path.getsize(path) != f.size:
            raise ModelError(f"{f.path} has {os.path.getsize(path)} bytes, expected {f.size}")
        if sha256_of(path) != f.sha256:
            raise ModelError(f"{f.path} does not match its sha256")


def free_bytes(path: str) -> int:
    """Free bytes on the disk that holds ``path``."""
    return shutil.disk_usage(path).free


def ensure(checkpoints: str, sources: Sequence[Source], report: Callable[[int, int, bool], None],
           list_files: Callable[[Source], List[RemoteFile]] = remote_files,
           download: Callable[[Source, str], None] = fetch, interval: float = 1.0,
           disk_free: Callable[[str], int] = free_bytes) -> List[str]:
    """Download and check every source that has no marker; returns the names fetched.

    ``report(done, total, complete)`` receives the bytes present out of the bytes needed, at
    start and about every ``interval`` seconds while something changes, with ``complete``
    False; then once with ``complete`` True after every source is checked and marked.

    Raises:
        ModelError: If the disk lacks room for the missing bytes plus ``SPARE_BYTES`` (nothing
            is downloaded), or a source's files do not match after the download (no marker
            is written).
    """
    todo = [s for s in sources if not os.path.isfile(marker_path(checkpoints, s))]
    if not todo:
        return []
    plan = [(s, target_dir(checkpoints, s), list_files(s)) for s in todo]
    total = sum(f.size for _s, _t, files in plan for f in files)

    def present() -> int:
        return min(total, sum(bytes_present(t, files) for _s, t, files in plan))

    os.makedirs(checkpoints, exist_ok=True)
    needed, free = total - present() + SPARE_BYTES, disk_free(checkpoints)
    if free < needed:
        raise ModelError(f"the models need {needed / 1e9:.1f} GB free on this disk, "
                         f"{free / 1e9:.1f} GB are available")

    stop = threading.Event()

    def poll() -> None:
        last = None
        while not stop.wait(interval):
            done = present()
            if done != last:
                report(done, total, False)
                last = done

    report(present(), total, False)
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        for source, target, files in plan:
            os.makedirs(target, exist_ok=True)
            download(source, target)
            check(target, files)
            os.makedirs(os.path.dirname(marker_path(checkpoints, source)), exist_ok=True)
            with open(marker_path(checkpoints, source), "w", encoding="utf-8") as fh:
                json.dump({"repo": source.repo, "revision": source.revision, "files": len(files)}, fh)
    finally:
        stop.set()
        poller.join()
    report(total, total, True)
    return [s.name for s in todo]

"""Remove build paths from trusted TorchScript checkpoints and audit a model directory.

Pass each build prefix as `--replace OLD NEW` and each forbidden expression as
`--forbid REGEX`. With no replacements the command only verifies. Run with a
CPU-capable PyTorch installation. Only *_compiled.model debug records are edited.
"""

import argparse
import copy
import io
import os
from pathlib import Path
import pickle
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch
from torch.jit._pickle import build_intlist


class IntList(list):
    """Preserve TorchScript's typed lists across Python pickle round trips."""

    def __reduce__(self):
        return build_intlist, (list(self),)


class DebugUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch.jit._pickle", "build_intlist"):
            return IntList
        raise pickle.UnpicklingError(f"Unexpected debug pickle global {module}.{name}")


def replace_debug_paths(data, replacements):
    marker, table, records = DebugUnpickler(io.BytesIO(data)).load()
    if marker != "FORMAT_WITH_STRING_TABLE":
        raise ValueError(f"Unknown TorchScript debug format {marker!r}")
    updated = []
    for line in table:
        for source, target in replacements:
            line = line.replace(source, target)
        updated.append(line)
    if tuple(updated) == table:
        return data
    # Protocol 2 is understood by TorchScript's C++ unpickler.
    return pickle.dumps((marker, tuple(updated), records), protocol=2)


def check_content(data, forbidden, location):
    if forbidden and forbidden.search(data):
        raise ValueError(f"Forbidden pattern found in {location}")


def verify_archive(path, forbidden):
    debug_count = 0
    check_content(path.read_bytes(), forbidden, path)
    with ZipFile(path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Invalid ZIP checksum in {path} ({damaged})")
        for member in archive.infolist():
            data = archive.read(member)
            location = f"{path} ({member.filename})"
            check_content(member.filename.encode(), forbidden, location)
            check_content(data, forbidden, location)
            if member.filename.endswith(".debug_pkl"):
                pickle.loads(data)
                debug_count += 1
    return debug_count


def load_compiled(path):
    model = torch.jit.load(str(path), map_location="cpu")
    # Reading source ranges exercises the lazy C++ debug unpickler as well.
    for module in model.modules():
        for method in module._c._method_names():
            for node in module._c._get_method(method).graph.nodes():
                node.sourceRange()
    return model.state_dict()


def check_weights(original, updated):
    if original.keys() != updated.keys():
        raise ValueError("Changed state_dict keys")
    for name, left in original.items():
        right = updated[name]
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"Changed dtype or shape for {name}")
        left_bytes = left.contiguous().reshape(-1).view(torch.uint8)
        right_bytes = right.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left_bytes, right_bytes):
            raise ValueError(f"Changed tensor bytes for {name}")


def process_compiled(path, replacements, forbidden):
    """Stage beside the original, compare the weights and replace atomically."""
    original = load_compiled(path)
    if not replacements:
        return verify_archive(path, forbidden), 0, 0

    changed = 0
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / path.name
        with ZipFile(path) as source, ZipFile(candidate, "w") as target:
            target.comment = source.comment
            for info in source.infolist():
                data = source.read(info)
                if info.filename.endswith(".debug_pkl"):
                    updated = replace_debug_paths(data, replacements)
                    changed += updated != data
                    data = updated
                target.writestr(copy.copy(info), data)
        debug_count = verify_archive(candidate, forbidden)
        check_weights(original, load_compiled(candidate))
        if changed:
            os.replace(candidate, path)
    return debug_count, changed, len(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="Directory containing the .model files")
    parser.add_argument("--replace", nargs=2, action="append", default=[],
                        metavar=("OLD", "NEW"), help="Literal prefix and replacement")
    parser.add_argument("--forbid", action="append", default=[], metavar="REGEX",
                        help="Reject matches in raw archives and decompressed members (case-insensitive)")
    args = parser.parse_args()
    if any(not source for source, _ in args.replace):
        parser.error("A replacement source must be nonempty")
    patterns = [f"(?:{pattern})".encode() for pattern in args.forbid]
    forbidden = re.compile(b"|".join(patterns), re.IGNORECASE) if patterns else None
    paths = sorted(args.models.rglob("*.model"))
    if not paths:
        parser.error("No .model files found")

    debug = changed = compiled = tensors = 0
    for path in paths:
        check_content(path.relative_to(args.models).as_posix().encode(), forbidden, path)
        if path.name.endswith("_compiled.model"):
            records, edits, compared = process_compiled(path, args.replace, forbidden)
            debug += records
            changed += edits
            tensors += compared
            compiled += 1
        else:
            debug += verify_archive(path, forbidden)
    print(f"Verified {len(paths)} model archives, {compiled} TorchScript models, "
          f"{debug} debug records ({changed} edited), {tensors} state tensors compared. "
          f"Forbidden-pattern hits: 0 ({len(patterns)} expressions).")


if __name__ == "__main__":
    main()

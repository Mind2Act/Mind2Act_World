"""Remove stale exporter documentation from a derived robot/camera layer.

This offline operation preserves every other USD opinion, including asset
references, render bindings, transforms and physics. Original assets are read
only. Old documentation belongs in the private build receipt, not the release.
"""
from pathlib import Path
import hashlib
import re

MACHINE_PATH = re.compile(r"/(?:home|root|shared|Users)/[^\s\"<>]+")


def without_export_history(source, destination):
    from pxr import Sdf

    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or source == destination.resolve():
        raise ValueError("export-history cleanup requires a fresh derived layer")
    original = Sdf.Layer.OpenAsAnonymous(str(source))
    if original is None or not MACHINE_PATH.search(original.documentation):
        raise ValueError("layer has no machine-specific exporter documentation")
    documentation = original.documentation
    original.documentation = ""
    content = original.ExportToString()
    if re.search(r"/(?:home|Users|shared/users)/|@/(?:root)/|/(?:root)/(?:share|workspace[^/]*|\.cache)/", content):
        raise ValueError("machine path outside layer documentation requires separate review")
    digest = hashlib.sha256(content.encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.Export(str(destination))
    reopened = Sdf.Layer.OpenAsAnonymous(str(destination))
    if reopened is None or reopened.documentation or reopened.ExportToString() != content:
        raise ValueError("export-history cleanup changed another USD opinion")
    return dict(operation="clear stale layer documentation", non_documentation_sha256=digest,
                removed_documentation_sha256=hashlib.sha256(documentation.encode()).hexdigest(),
                removed_machine_path_count=len(MACHINE_PATH.findall(documentation)))

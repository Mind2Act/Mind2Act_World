"""Stored, 64-byte-aligned USDZ fixture writer, including Isaac USD 24.5."""
from contextlib import contextmanager
from pathlib import Path
import struct
import zipfile


@contextmanager
def package_writer(path):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        class Writer:
            def AddFile(self, source, name):
                info = zipfile.ZipInfo(name)
                padding = -(archive.fp.tell() + 30 + len(name.encode('utf-8'))) % 64
                if padding:
                    if padding < 4:
                        padding += 64
                    info.extra = b'\xff\xff' + struct.pack('<H', padding - 4) + bytes(padding - 4)
                archive.writestr(info, Path(source).read_bytes())
                return name
        yield Writer()

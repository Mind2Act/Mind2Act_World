"""CP06 append-only physics audit; evidence snapshots contain only receipts."""
import hashlib
import json
from pathlib import Path


class Audit:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('xb')
        self.digest = hashlib.sha256()
        self.rows = self.bytes = 0

    def append(self, row):
        if row['tick'] != self.rows:
            raise ValueError('CP06 audit requires consecutive physics ticks including initial state')
        data = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        self.stream.write(data)
        self.digest.update(data)
        self.bytes += len(data)
        self.rows += 1

    def receipt(self):
        self.stream.flush()
        return dict(path=str(self.path), sha256=self.digest.hexdigest(), rows=self.rows, bytes=self.bytes)

    def close(self):
        self.stream.close()


def read(receipt, tick):
    path = Path(receipt['path'])
    if (not path.is_absolute() or type(receipt['bytes']) is not int
            or receipt['bytes'] <= 0 or receipt['rows'] != tick + 1):
        raise ValueError('CP06 invalid audit receipt')
    data = path.read_bytes()
    if len(data) != receipt['bytes'] or hashlib.sha256(data).hexdigest() != receipt['sha256']:
        raise ValueError('CP06 audit digest/length mismatch')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != tick+1 or any(type(r['tick']) is not int or r['tick'] != i for i,r in enumerate(rows)):
        raise ValueError('CP06 audit dropped/repeated physics sample')
    return rows

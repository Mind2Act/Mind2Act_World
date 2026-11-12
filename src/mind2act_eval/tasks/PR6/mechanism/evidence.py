"""Native pose/contact parsing selected from source; no cross-task rules."""
from dataclasses import dataclass
import math

class EvidenceError(ValueError):
    pass

def vector(value, n, label):
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise EvidenceError(f'{label}: expected {n} values')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise EvidenceError(f'{label}: non-finite/non-numeric measurement')
    return tuple(float(v) for v in value)

@dataclass(frozen=True)
class SupportContact:
    path: str
    position: tuple
    normal: tuple
    separation: float

    @classmethod
    def parse(cls,row,contacts):
        if row['path'] not in contacts:raise EvidenceError('Support point is not an active contact')
        normal=vector(row['normal'],3,'contact normal')
        if not .99<=math.dist(normal,(0,0,0))<=1.01:raise EvidenceError('Contact normal must be unit length')
        separation=vector([row['separation']],1,'contact separation')[0]
        return cls(row['path'],vector(row['position'],3,'contact position'),normal,separation)

@dataclass(frozen=True)
class Body:
    lower: tuple
    upper: tuple
    linear: tuple
    angular: tuple
    contacts: tuple[str, ...]
    supports: tuple[SupportContact, ...] | None = None

    @classmethod
    def parse(cls, row):
        lo, hi = vector(row['lower'], 3, 'lower'), vector(row['upper'], 3, 'upper')
        if any(a >= b for a, b in zip(lo, hi)):
            raise EvidenceError('degenerate body bounds')
        contacts = row['contacts']
        if not isinstance(contacts, list) or any(type(x) is not str for x in contacts):
            raise EvidenceError('contacts must be measured path strings')
        supports=None
        if 'support_contacts' in row:
            if not isinstance(row['support_contacts'],list):raise EvidenceError('support_contacts must be a list')
            supports=tuple(SupportContact.parse(v,contacts) for v in row['support_contacts'])
        return cls(lo, hi, vector(row['linear'], 3, 'linear'), vector(row['angular'], 3, 'angular'), tuple(contacts),supports)

    @property
    def center(self):
        return tuple((a+b)/2 for a, b in zip(self.lower, self.upper))

    @property
    def released(self):
        return not any(x.startswith('/World/Robot') for x in self.contacts)

    def quiet(self, config):
        return (math.dist(self.linear, (0, 0, 0)) <= config['max_linear_speed'] and
                math.dist(self.angular, (0, 0, 0)) <= config['max_angular_speed'])


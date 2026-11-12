"""One request to an explicitly provisioned original native public planner."""
import json
from math import isfinite
import urllib.request
from urllib.parse import urlparse

import numpy as np


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RemotePlanner:
    def __init__(self, url, timeout_s):
        parts = urlparse(url)
        if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1") or parts.username or parts.password:
            raise ValueError("native public planner requires an explicit loopback HTTP endpoint")
        if isinstance(timeout_s, bool) or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("explicit positive planner timeout required")
        self.url, self.timeout = url, timeout_s
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def __call__(self, *, arm, q, target_pose):
        q, pose = np.asarray(q, dtype=float), np.asarray(target_pose, dtype=float)
        if arm not in ("left", "right") or q.shape != (6,) or pose.shape != (7,) or not np.isfinite(np.r_[q, pose]).all():
            raise ValueError("public planner expects arm, six joint radians and xyz+wxyz goal")
        request = urllib.request.Request(self.url,
            data=json.dumps({"arm": arm, "q": q.tolist(), "target_pose": pose.tolist()}, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"})
        with self.http.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("status") == "Error":
            raise RuntimeError("native public planner infrastructure failure")
        return result

    def close(self):
        pass


def create(config):
    if not isinstance(config, dict) or set(config) != {"url", "timeout_s"}:
        raise ValueError("explicit native planner url and timeout_s required")
    return RemotePlanner(**config)

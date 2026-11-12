"""Load the unchanged, pinned official XPolicyLab WebSocket implementation.

This module adds no wire protocol. Install requirements-ws.txt or pass a checkout
of the official pin. A loaded CodingAgent package with the same name is rejected.
"""
from __future__ import annotations
import hashlib
import importlib
import os
from pathlib import Path
import sys

XPOLICY_COMMIT = "128e626278ef753fa8a86b16e3425a6359388b4c"
PROTOCOL_VERSION = "xpolicylab-ws@" + XPOLICY_COMMIT
_SOURCE_HASHES = {'XPolicyLab.py': 'a3909558697b73479f8baee03e15b4d74060a7c9ac2f31fcbc6cd659e7364862',
 'client_server/ws/__init__.py': '79905987a12ebf318dc0b2b9ec6f375d9b4561bac29d6c27d04e318d8f6e00b3',
 'client_server/ws/__main__.py': 'ee1c5807dcdf1c8f6721c5447463eb51babdb0f9ffbd91d1e7343834ca4aa19a',
 'client_server/ws/model_client.py': '4d6e225ae5f577d862366350dca8742e91dd9527d6e76b8b2a6d48c991212d67',
 'client_server/ws/model_server.py': '4988823b436cd05e6a4d8b13012cc30c7ffd57711b72497b1223a6b10dc0c043',
 'client_server/ws/protocol/__init__.py': '19aeae375e94e58d437d789d0aff07b232c13f375235384d784984631d25d5d9',
 'client_server/ws/protocol/client.py': 'b309f58b1bddd99e56578026b61a15e6cfdc07e843e8a082bfd291d0e853ce5a',
 'client_server/ws/protocol/codec.py': '8a97ef7e102fd954c266352d81eaceef0f1b0f5504571c03037683efe387865a',
 'client_server/ws/protocol/exceptions.py': '73161f4cdcc7ffc8a87cec94f5459bfcb9113076be272d6cb2a3ff8d69e0c6ce',
 'client_server/ws/protocol/keepalive.py': '7f6b9e636eefcd6c80a4b62e404a6f89e2f7e1a3b6008500312a972d6990a21d',
 'client_server/ws/protocol/messages.py': 'd05c8cbc20e0cbe1502551864907f25897035993f920748c9290c944c959eab7',
 'client_server/ws/protocol/schemas.py': 'f1ca82eafa8dcad6d968508adb51c25342b291f7b1c553fff305c381a6098bd0',
 'utils/load_file.py': '488590d76084f419a36c2fb3e658eeb240ae554678899e9f2ce0cc10da5c4b57',
 'utils/process_data.py': '593b72c64b5a897f2388e6a937d08ff4fc22b51af3bc0a8a0d2b221c235020bd'}


def load_xpolicy(source=None):
    source = source or os.environ.get("XPOLICYLAB_SOURCE_ROOT")
    if source:
        sys.path.insert(0, str(Path(source).resolve()))
    module = importlib.import_module("client_server.ws.model_client")
    root = Path(module.__file__).resolve().parents[2]
    for relative, expected in _SOURCE_HASHES.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Official XPolicyLab pin differs: {path}")
    # Catch an already imported same-name package from CodingAgent or another tree.
    image_module = importlib.import_module("XPolicyLab.utils.process_data")
    if Path(image_module.__file__).resolve() != root / "utils/process_data.py":
        raise RuntimeError("XPolicyLab image codec was imported from a different source")
    server = importlib.import_module("client_server.ws.model_server")
    return module.WsModelClient, server.PolicyServer, server.PolicyServerConfig


def create_client(url, *, evaluation_id="mind2act", trial_id="episode", source=None, **kwargs):
    client, _, _ = load_xpolicy(source)
    return client(url=url, evaluation_id=evaluation_id, trial_id=trial_id, **kwargs)

"""Private original/test protocols are unavailable in public runtime code."""
WORLD_SHA256 = 'ef84be71411f69b0a7dde111ecec713222ebbbca623bb0eea6dbbb727961bd6d'
def _blocked(*args, **kwargs):
    raise ValueError("public runtime requires an exact selected training identity")
validate_binding = authenticate = _blocked

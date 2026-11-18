"""Single-request native HTTP worker; stdout is a private parent pipe."""
import json
import os
import sys
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    payload = json.load(sys.stdin)
    key = os.environ[payload["key_env"]]
    request = urllib.request.Request(payload["url"],
        data=json.dumps(payload["request"], allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=payload["timeout_s"]) as response:
            result = {"status": response.status, "response": json.load(response)}
    except urllib.error.HTTPError as error:
        body = error.read(65536).decode("utf-8", errors="replace")
        result = {"status": error.code, "error_body": body.replace(key, "<key>")[:2000],
                  "retry_after": error.headers.get("Retry-After")}
    except Exception as error:
        result = {"error_type": type(error).__name__}
    json.dump(result, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()

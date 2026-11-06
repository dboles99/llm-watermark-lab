"""Run the simulator on loopback with a single in-memory nonce authority."""

from __future__ import annotations

import uvicorn

from watermark_lab.tee_server.app import app


def main() -> None:
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8787,
        access_log=False,
        server_header=False,
        workers=1,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

from app import app


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

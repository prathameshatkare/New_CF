"""Launch FastAPI web UI for CF risk screening."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8600, reload=False)


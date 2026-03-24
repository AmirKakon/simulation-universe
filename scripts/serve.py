"""Quick-start script for the SimVerse API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "simverse.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

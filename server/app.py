import os
import uvicorn
from src.server import app

def main():
    """Entry point for the server script required by OpenEnv."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.server:app", host=host, port=port)

if __name__ == "__main__":
    main()

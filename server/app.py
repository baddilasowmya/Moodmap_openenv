"""
Server entry point for MoodMap OpenEnv deployment.
"""
from app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
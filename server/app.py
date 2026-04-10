"""
Server entry point for MoodMap OpenEnv deployment.
"""
import sys
import os

# Add project root so top-level modules (app, graders, moodmap_env) are importable.
# Must happen before any import of those modules.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Remove 'server' package itself from sys.modules so that when app.py does
# 'from graders import ...' Python looks up graders fresh as a top-level
# package rather than resolving it relative to a partially-loaded server.app.
sys.modules.pop("server", None)
sys.modules.pop("server.app", None)

# Now safe to import — graders/__init__.py will be fully initialized
# before anything tries to pull names from it.
from app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
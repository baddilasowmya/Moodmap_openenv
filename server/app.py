"""
Server entry point for MoodMap OpenEnv deployment.
"""
import sys
import os

# Add project root to sys.path so top-level modules (app, graders, moodmap_env) are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    import uvicorn
    # Point directly at app:app — avoids re-importing app through server.app,
    # which registers the same module under two names and causes a
    # circular/partial-initialization ImportError in graders.
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
"""Entry point for the packaged (PyInstaller) game executable.

Keeps the exe build simple: it just calls the play server's main().
For development, prefer:  python -m complete_ai.play_server
"""

from complete_ai.play_server import main

if __name__ == "__main__":
    main()

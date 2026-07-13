# PyInstaller spec — build a distributable "ユビスマ対戦" executable.
#
# Build:  pip install pyinstaller
#         pyinstaller yubisuma_game.spec
# Output: dist/ユビスマ対戦/  (onedir; zip the whole folder for GitHub Releases)
#
# Notes:
#  - onedir (not onefile): torch/numba unpack faster and are more reliable.
#  - Uses whatever torch is installed. For a smaller, portable build install
#    CPU-only torch first:  pip install torch --index-url https://download.pytorch.org/whl/cpu
#  - The value-net model and the web UI are bundled via `datas`.
#  - Test the result on a machine WITHOUT Python before publishing.

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden = (
    collect_submodules("complete_solver")
    + collect_submodules("complete_ai")
    + collect_submodules("numba")
    + collect_submodules("llvmlite")
    + ["scipy.optimize", "scipy._lib.messagestream"]
)

datas = [
    ("complete_ai/webplay", "webplay"),
    ("models/value_latest.pt", "models"),
]
# Numba needs its compiled extension data files present at runtime.
datas += collect_data_files("numba")
datas += collect_data_files("llvmlite")

a = Analysis(
    ["play_entry.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter", "PyQt5", "PySide2", "wandb",
              "sb3_contrib", "stable_baselines3", "gymnasium"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ユビスマ対戦",
    console=True,          # keep a console so load progress is visible
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ユビスマ対戦",
)

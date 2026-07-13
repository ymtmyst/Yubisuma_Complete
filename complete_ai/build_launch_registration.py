"""Generate the one-time setup that lets the report hub launch the game.

Browsers cannot run a .bat from a file:// link (security). The standard way
around this is a custom URL protocol: register ``yubisuma://`` once, and then
a normal link ``<a href="yubisuma://play">`` in the hub launches the game.

This writes two double-clickable files at the Complete root:
  - AI対戦リンクを有効化.reg   (register the protocol → run AIと対戦.bat)
  - AI対戦リンクを無効化.reg   (remove it)

.reg files must be UTF-16 LE with a BOM, and Windows paths need doubled
backslashes and escaped quotes — handled here so the ★ / Japanese path is
written correctly.

Run:  python -m complete_ai.build_launch_registration
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPLETE = Path(__file__).resolve().parent.parent
BAT = COMPLETE / "AIと対戦.bat"


def _reg_escape(path: str) -> str:
    # In .reg string values: backslash and quote are escaped as \\ and \".
    return path.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bat_esc = _reg_escape(str(BAT))
    # cmd /c "<bat>"  — the bat ignores the trailing yubisuma://play arg.
    command = f'cmd /c \\"{bat_esc}\\"'

    enable = (
        "Windows Registry Editor Version 5.00\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma]\r\n"
        '@="URL:Yubisuma Play Protocol"\r\n'
        '"URL Protocol"=""\r\n\r\n'
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell\\open]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell\\open\\command]\r\n"
        f'@="{command}"\r\n'
    )
    disable = (
        "Windows Registry Editor Version 5.00\r\n\r\n"
        "[-HKEY_CURRENT_USER\\Software\\Classes\\yubisuma]\r\n"
    )

    enable_path = COMPLETE / "AI対戦リンクを有効化.reg"
    disable_path = COMPLETE / "AI対戦リンクを無効化.reg"
    # UTF-16 LE with BOM is what regedit expects.
    enable_path.write_text(enable, encoding="utf-16")
    disable_path.write_text(disable, encoding="utf-16")

    print(f"生成しました:\n  {enable_path.name}\n  {disable_path.name}")
    print(f"登録コマンド: {command}")
    print("使い方: 有効化.reg をダブルクリック → 「はい」で登録 →"
          " ハブの『▶ AIと対戦』が使えるようになります。")


if __name__ == "__main__":
    main()

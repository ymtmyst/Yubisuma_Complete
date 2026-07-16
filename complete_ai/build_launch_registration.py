"""Generate the one-time setup that lets the report hub launch the game.

Browsers cannot run a .bat from a file:// link (security). The standard way
around this is a custom URL protocol. This registers both launchers:

  - ``yubisuma://play``       -> current Complete rules
  - ``yubisuma-beta://play``  -> isolated v1 beta rules

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
ROOT = COMPLETE.parent
BAT = COMPLETE / "AIと対戦.bat"
BETA_BAT = ROOT / "Beta_v1" / "play_beta.bat"


def _reg_escape(path: str) -> str:
    # In .reg string values: backslash and quote are escaped as \\ and \".
    return path.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bat_esc = _reg_escape(str(BAT))
    beta_bat_esc = _reg_escape(str(BETA_BAT))
    command = f'cmd /c \\"{bat_esc}\\"'
    beta_command = f'cmd /c \\"{beta_bat_esc}\\"'

    enable = (
        "Windows Registry Editor Version 5.00\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma]\r\n"
        '@="URL:Yubisuma Play Protocol"\r\n'
        '"URL Protocol"=""\r\n\r\n'
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell\\open]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma\\shell\\open\\command]\r\n"
        f'@="{command}"\r\n\r\n'
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma-beta]\r\n"
        '@="URL:Yubisuma Beta Play Protocol"\r\n'
        '"URL Protocol"=""\r\n\r\n'
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma-beta\\shell]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma-beta\\shell\\open]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\yubisuma-beta\\shell\\open\\command]\r\n"
        f'@="{beta_command}"\r\n'
    )
    disable = (
        "Windows Registry Editor Version 5.00\r\n\r\n"
        "[-HKEY_CURRENT_USER\\Software\\Classes\\yubisuma]\r\n\r\n"
        "[-HKEY_CURRENT_USER\\Software\\Classes\\yubisuma-beta]\r\n"
    )

    enable_path = COMPLETE / "AI対戦リンクを有効化.reg"
    disable_path = COMPLETE / "AI対戦リンクを無効化.reg"
    # UTF-16 LE with BOM is what regedit expects.
    enable_path.write_text(enable, encoding="utf-16")
    disable_path.write_text(disable, encoding="utf-16")

    print(f"生成しました:\n  {enable_path.name}\n  {disable_path.name}")
    print(f"現行版: {command}")
    print(f"ベータ版: {beta_command}")
    print("使い方: 有効化.reg をダブルクリック → 「はい」で登録 →"
          " ハブの2つの対戦ボタンが使えるようになります。")


if __name__ == "__main__":
    main()

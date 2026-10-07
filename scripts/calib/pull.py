"""Copy calibration recordings off the iPhone (app Documents/calibration/<name>) over USB.

usage: python pull.py <local-dir> [name]      (newest recording when name is omitted)
Run with ~/calib/venv/bin/python in WSL; talks to usbmuxd (our Apple Devices relay).
"""
import asyncio
import sys
from pathlib import Path

from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.house_arrest import HouseArrestService

BUNDLE = "XTL-4YXF2BUX87.com.weeesports.vr"   # xtool prefixes the bundle ID with the team


async def main(local: Path, name: str | None) -> None:
    lockdown = await create_using_usbmux()
    async with await HouseArrestService.create(lockdown=lockdown, bundle_id=BUNDLE, documents_only=True) as afc:
        recordings = sorted(n for n in await afc.listdir("/Documents/calibration") if n not in (".", ".."))
        if not recordings:
            sys.exit("no recordings in Documents/calibration")
        print("on phone:", ", ".join(recordings))
        name = name or recordings[-1]
        local.mkdir(parents=True, exist_ok=True)
        await afc.pull(f"/Documents/calibration/{name}", str(local))
        print(f"pulled {name} -> {local / name}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None))

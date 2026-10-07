"""Render Kalibr's AprilGrid (10 x 6 tag36h11, spacing 0.3) as tools/aprilgrid.svg for a monitor.

usage: python make_target.py <kalibr-repo> <out.svg>
Uses Kalibr's own generator (aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf), so the
layout and tag border match its detector; only the caption (which needs LaTeX) is skipped.
"""
import sys
from pathlib import Path

from pyx import canvas


def main(kalibr: Path, out: Path) -> None:
    source = (kalibr / "aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf").read_text()
    module: dict = {"__name__": "kalibr_target"}
    exec(compile(source, "kalibr_create_target_pdf", "exec"), module)
    c = canvas.canvas()
    module["c"] = c                                      # its tag drawing uses a global canvas `c`
    c.text = lambda *args, **kwargs: None                # caption only; needs LaTeX
    module["generateAprilBoard"](c, 10, 6, 1.0, 0.3, skipIds=[])    # unit tag size; the monitor sets the scale
    c.writeSVGfile(str(out.with_suffix("")))
    # Each tag bit is its own rectangle; anti-aliased seams between them would show as grey lines.
    svg = out.read_text().replace("<svg ", '<svg shape-rendering="crispEdges" ', 1)
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

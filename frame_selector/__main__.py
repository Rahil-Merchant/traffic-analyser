"""Pick the frames of one or more videos most worth labelling.

    python -m frame_selector gui                                          # local web page (recommended)
    python -m frame_selector VIDEO [VIDEO ...] --model premier_station.engine --out mined/
    python -m frame_selector VIDEO --detections dump.npz --out mined/     # reuse a detection dump (no model needed)

Output: mined/images/*.jpg (full resolution), mined/labels/*.txt (YOLO pseudo-labels to correct),
classes.txt, manifest.csv (why each frame was chosen), review.html (thumbnails grouped by reason).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['gui']:
        from frame_selector.gui import serve
        ap = argparse.ArgumentParser(prog='frame_selector gui')
        ap.add_argument('--port', type=int, default=8090)
        ap.add_argument('--no-browser', action='store_true')
        a = ap.parse_args(argv[1:])
        return serve(a.port, open_browser=not a.no_browser)
    from frame_selector.pipeline import run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='+')
    ap.add_argument('--model', help='.pt or .engine; needed unless --detections is given for every video')
    ap.add_argument('--detections', nargs='*', default=[], help='existing detection dumps, one per video in order')
    ap.add_argument('--out', default='mined')
    ap.add_argument('--budget', type=int, default=100, help='frames per 2 hours of video')
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)
    try:
        run(a.videos, a.out, a.model, a.detections, a.budget, a.device)
    except ValueError as e:
        ap.error(str(e))


if __name__ == '__main__':
    main()

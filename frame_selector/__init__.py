"""Frame selector: picks the frames of a video most worth labelling.

Without labels, a mistake has to be inferred. A vehicle is one object across hundreds of frames,
so a detection that disagrees with its own track (missing, extra, or a different class) is probably
wrong. Only the model's own output is used: its tracks, its low-confidence detections and its
class votes. See `python -m frame_selector --help`.
"""

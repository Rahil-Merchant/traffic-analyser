"""Stage 4: pick a diverse set of frames within the budget."""
from collections import defaultdict

QUOTAS = {'fn': 0.30, 'fp': 0.25, 'class': 0.30, 'borderline': 0.15}
MIN_SPACING_S = 3.0


def budget_for(duration_s, per_two_hours=100):
    return max(20, round(per_two_hours * duration_s / 7200.0))


def select_frames(candidates, budget, fps):
    """-> list of (frame, [candidates on that frame]) sorted by frame. Each chosen frame carries every
    candidate flagged on it, so the labeller sees all of the frame's issues at once."""
    best = {}                                   # one candidate per (track, reason); orphans keyed by frame
    for c in candidates:
        key = (c['track'] if c['track'] is not None else ('f', c['frame']), c['reason'])
        if key not in best or c['score'] > best[key]['score']:
            best[key] = c
    per_frame = defaultdict(list)
    for c in best.values():
        per_frame[c['frame']].append(c)
    pools = defaultdict(list)
    for f, cs in per_frame.items():
        top = max(cs, key=lambda c: c['score'])
        pools[top['reason']].append((top['score'] + 0.1 * (len(cs) - 1), f))
    for r in pools:
        pools[r].sort(reverse=True)
    spacing = int(MIN_SPACING_S * fps)
    chosen = []

    def far_enough(f):
        return all(abs(f - g) >= spacing for g in chosen)

    quota = {r: int(round(budget * q)) for r, q in QUOTAS.items()}
    for r, n in quota.items():
        taken = 0
        for _, f in pools[r]:
            if taken >= n:
                break
            if far_enough(f):
                chosen.append(f)
                taken += 1
    leftovers = sorted((s, f) for r in pools for s, f in pools[r] if f not in chosen)
    for _, f in reversed(leftovers):
        if len(chosen) >= budget:
            break
        if far_enough(f):
            chosen.append(f)
    return [(f, sorted(per_frame[f], key=lambda c: -c['score'])) for f in sorted(chosen)]

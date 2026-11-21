"""Helpers for reading and editing the correspondence list.

Sides are addressed with the strings ``'CAGE'`` and ``'TARGET'`` everywhere in
the add-on; :func:`points` maps a side to the matching point collection.
"""

import colorsys

SIDES = ('CAGE', 'TARGET')


def other(side):
    return 'TARGET' if side == 'CAGE' else 'CAGE'


def settings(context):
    return context.scene.mesh_wrap


def obj_for(mw, side):
    return mw.cage if side == 'CAGE' else mw.target


def points(pair, side):
    return pair.cage_points if side == 'CAGE' else pair.target_points


def coords(pair, side):
    return [tuple(p.co) for p in points(pair, side)]


def set_coords(pair, side, cos):
    coll = points(pair, side)
    coll.clear()
    for co in cos:
        coll.add().co = co


# ---------------------------------------------------------------- status ---

def has_side(pair, side):
    return len(points(pair, side)) > 0


def is_point_pair(pair):
    """A single landmark on each side (or on the one side drawn so far)."""
    counts = [len(pair.cage_points), len(pair.target_points)]
    drawn = [c for c in counts if c]
    return bool(drawn) and all(c == 1 for c in drawn)


def is_complete(pair):
    n_c, n_t = len(pair.cage_points), len(pair.target_points)
    if not n_c or not n_t:
        return False
    # a point on one side must be matched by a point on the other
    return (n_c == 1) == (n_t == 1)


def missing_side(pair):
    """The side still to be drawn, or None when both exist."""
    if not pair.cage_points:
        return 'CAGE'
    if not pair.target_points:
        return 'TARGET'
    return None


def sample_count(pair):
    return 1 if is_point_pair(pair) else max(2, pair.samples)


def counts(mw):
    complete = sum(1 for p in mw.pairs if is_complete(p))
    return complete, len(mw.pairs) - complete


def matched_sample_total(mw):
    return sum(sample_count(p) for p in mw.pairs
               if p.enabled and is_complete(p))


def active_pair(mw):
    if 0 <= mw.pair_index < len(mw.pairs):
        return mw.pairs[mw.pair_index]
    return None


def index_of_uid(mw, uid):
    for i, p in enumerate(mw.pairs):
        if p.uid == uid:
            return i
    return -1


def by_uid(mw, uid):
    i = index_of_uid(mw, uid)
    return mw.pairs[i] if i >= 0 else None


# ------------------------------------------------------------- creation ---

_GOLDEN = 0.6180339887498949


def palette_color(k):
    """Distinct, saturated colour for the k-th pair (golden-ratio hues)."""
    hue = (0.08 + k * _GOLDEN) % 1.0
    sat = 0.62 if k % 2 else 0.74
    return colorsys.hsv_to_rgb(hue, sat, 0.98)


def new_pair(mw, name=None):
    mw.uid_counter += 1
    uid = mw.uid_counter
    pair = mw.pairs.add()
    pair.uid = uid
    pair.name = name or f"Pair {uid}"
    pair.color = palette_color(uid - 1)
    pair.samples = mw.default_samples
    mw.pair_index = len(mw.pairs) - 1
    return pair


def remove(mw, index):
    if not 0 <= index < len(mw.pairs):
        return
    mw.pairs.remove(index)
    if mw.pair_index >= len(mw.pairs):
        mw.pair_index = len(mw.pairs) - 1
    elif mw.pair_index > index:
        mw.pair_index -= 1


# ------------------------------------------------------------ snapshots ---
# Used by the curve editor's private undo history.

_FIELDS = ('uid', 'name', 'closed', 'samples', 'weight', 'hidden', 'enabled')


def snapshot(mw):
    data = []
    for p in mw.pairs:
        rec = {f: getattr(p, f) for f in _FIELDS}
        rec['color'] = tuple(p.color)
        rec['CAGE'] = coords(p, 'CAGE')
        rec['TARGET'] = coords(p, 'TARGET')
        data.append(rec)
    return {'pairs': data, 'index': mw.pair_index,
            'uid_counter': mw.uid_counter}


def restore(mw, snap):
    mw.pairs.clear()
    for rec in snap['pairs']:
        p = mw.pairs.add()
        for f in _FIELDS:
            setattr(p, f, rec[f])
        p.color = rec['color']
        set_coords(p, 'CAGE', rec['CAGE'])
        set_coords(p, 'TARGET', rec['TARGET'])
    mw.uid_counter = max(mw.uid_counter, snap['uid_counter'])
    mw.pair_index = min(snap['index'], len(mw.pairs) - 1)

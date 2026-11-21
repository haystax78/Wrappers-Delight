"""Convert correspondences saved by earlier versions of this add-on.

Old files store one record per matched point in ``mesh_wrap.landmarks``;
records drawn as one stroke share a ``curve_id`` (-1 = standalone point).
Each stroke becomes one :class:`CurvePair`, each standalone point a point
pair.  Coordinates already use the same convention (cage local rest shape,
target local space), so they are copied unchanged.
"""

from . import pairs as P


def has_legacy(mw):
    return len(mw.landmarks) > 0


def convert(mw):
    """Append converted pairs; returns how many were created."""
    groups = {}
    order = []
    singles = []
    for rec in mw.landmarks:
        if rec.curve_id < 0:
            singles.append(rec)
            continue
        if rec.curve_id not in groups:
            groups[rec.curve_id] = []
            order.append(rec.curve_id)
        groups[rec.curve_id].append(rec)

    made = 0
    for cid in order:
        recs = groups[cid]
        cage = [tuple(r.src_co) for r in recs if r.has_src]
        target = [tuple(r.tgt_co) for r in recs if r.has_tgt]
        if not cage and not target:
            continue
        pair = P.new_pair(mw)
        pair.closed = any(r.curve_closed for r in recs)
        pair.samples = max(2, min(256, max(len(cage), len(target))))
        P.set_coords(pair, 'CAGE', cage)
        P.set_coords(pair, 'TARGET', target)
        made += 1
    for rec in singles:
        if not (rec.has_src or rec.has_tgt):
            continue
        pair = P.new_pair(mw, name=f"Point {mw.uid_counter + 1}")
        if rec.has_src:
            P.set_coords(pair, 'CAGE', [tuple(rec.src_co)])
        if rec.has_tgt:
            P.set_coords(pair, 'TARGET', [tuple(rec.tgt_co)])
        made += 1
    mw.landmarks.clear()
    return made

"""
scale_fossil.py - scale a fossil endocast and its crest lines by a scale factor S that YOU choose,
before aligning it with the rigid script (registrationandlinesets4.py, rotation + translation only).

Every file is scaled about the SAME centre (the centroid of MESH), so mesh and curves stay together.
Only the x, y, z coordinates change; everything else in the files (faces, lines, labels, quality,
colours, other values) is copied unchanged. Output files get the suffix _S<value> next to the inputs.

Formats: PLY (binary little-endian or ASCII) and legacy VTK (ASCII or BINARY), as written by MeshLab,
Avizo, ParaView and the crest-line tools.

How to choose S:
  - S = the size factor you trust (e.g. from a published endocranial volume, a linear dimension, or
    centroid size), as  S = (target size / fossil size)  for a length, or (V_target / V_fossil)^(1/3) for volumes.
  - Or let the script suggest one from volumes: set REFERENCE to a CLOSED mesh of the target (e.g. the
    extant average surface) and leave S = None. This needs MESH to be closed too (not for open fragments).

Run:  python scale_fossil.py      (needs numpy only)
"""
import os
import numpy as np

# ----------------------------- SETTINGS -----------------------------
MESH = r"I:\path\to\fossil_endocast.ply"           # the fossil mesh (defines the centre)
CURVES = [                                          # crest-line files in the same frame (may be empty)
    r"I:\path\to\fossil_ridges.vtk",
    r"I:\path\to\fossil_ravines.vtk",
]
S = None                                            # your scale factor, e.g. 0.944; None = suggest from volumes
REFERENCE = r""                                     # closed target mesh, only used when S is None
# --------------------------------------------------------------------


# ---------------- PLY ----------------
PLY_TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2',
             'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
             'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}


def _ply_header(raw):
    end = raw.index(b'end_header') + len(b'end_header')
    end = raw.index(b'\n', end) + 1
    lines = raw[:end].decode('latin1').splitlines()
    fmt = next(l.split()[1] for l in lines if l.startswith('format'))
    elements, cur = [], None
    for l in lines:
        t = l.split()
        if not t: continue
        if t[0] == 'element':
            cur = [t[1], int(t[2]), []]; elements.append(cur)
        elif t[0] == 'property' and cur is not None:
            cur[2].append(t[1:])
    return end, fmt, elements


def _ply_vertices(raw):
    """Return (header_end, format, vertex_dtype_or_None, n_vertices, property names)."""
    end, fmt, elements = _ply_header(raw)
    if elements[0][0] != 'vertex':
        raise ValueError("PLY: the first element must be 'vertex'")
    name, n, props = elements[0]
    if any(p[0] == 'list' for p in props):
        raise ValueError("PLY: list properties on vertices are not supported")
    names = [p[1] for p in props]
    if fmt == 'binary_little_endian':
        dt = np.dtype([(p[1], '<' + PLY_TYPES[p[0]]) for p in props])
    elif fmt == 'ascii':
        dt = None
    else:
        raise ValueError(f"PLY format {fmt} not supported (save as binary little-endian or ASCII)")
    return end, fmt, dt, n, names


def read_ply_points(fn):
    raw = open(fn, 'rb').read()
    end, fmt, dt, n, names = _ply_vertices(raw)
    if dt is not None:
        v = np.frombuffer(raw, dtype=dt, count=n, offset=end)
        return np.c_[v['x'], v['y'], v['z']].astype(float)
    rows = raw[end:].decode('latin1').split('\n')[:n]
    ix = [names.index(c) for c in 'xyz']
    return np.array([[float(r.split()[i]) for i in ix] for r in rows])


def scale_ply(fn, out, S, c):
    raw = open(fn, 'rb').read()
    end, fmt, dt, n, names = _ply_vertices(raw)
    if dt is not None:
        v = np.frombuffer(raw, dtype=dt, count=n, offset=end).copy()
        for k, a in enumerate('xyz'):
            v[a] = (c[k] + S * (v[a].astype(float) - c[k])).astype(v.dtype[a])
        data = raw[:end] + v.tobytes() + raw[end + n * dt.itemsize:]
    else:
        body = raw[end:].decode('latin1').split('\n')
        ix = [names.index(a) for a in 'xyz']
        for r in range(n):
            t = body[r].split()
            for k, i in enumerate(ix):
                t[i] = repr(c[k] + S * (float(t[i]) - c[k]))
            body[r] = ' '.join(t)
        data = raw[:end] + '\n'.join(body).encode('latin1')
    open(out, 'wb').write(data)


# ---------------- legacy VTK ----------------
def _vtk_points_span(raw):
    """Locate the POINTS block: returns (binary?, n, dtype string, start, stop) in bytes."""
    binary = b'\nBINARY' in raw[:256]
    i = raw.index(b'POINTS')
    j = raw.index(b'\n', i)
    _, n, typ = raw[i:j].decode('latin1').split()[:3]
    return binary, int(n), typ, j + 1


def read_vtk_points(fn):
    raw = open(fn, 'rb').read()
    binary, n, typ, start = _vtk_points_span(raw)
    if binary:
        return np.frombuffer(raw, dtype='>f4' if typ == 'float' else '>f8', count=3 * n, offset=start).reshape(n, 3).astype(float)
    txt = raw[start:].decode('latin1').split()
    return np.array(txt[:3 * n], float).reshape(n, 3)


def scale_vtk(fn, out, S, c):
    raw = open(fn, 'rb').read()
    binary, n, typ, start = _vtk_points_span(raw)
    if binary:
        dt = '>f4' if typ == 'float' else '>f8'
        P = np.frombuffer(raw, dtype=dt, count=3 * n, offset=start).reshape(n, 3).astype(float)
        Q = (c + S * (P - c)).astype(dt)
        data = raw[:start] + Q.tobytes() + raw[start + Q.nbytes:]
    else:
        txt = raw[start:].decode('latin1')
        # walk tokens of the POINTS block, keeping everything after it byte for byte
        pos, vals = 0, []
        while len(vals) < 3 * n:
            while txt[pos].isspace(): pos += 1
            q = pos
            while q < len(txt) and not txt[q].isspace(): q += 1
            vals.append(float(txt[pos:q])); pos = q
        P = np.array(vals).reshape(n, 3)
        Q = c + S * (P - c)
        block = '\n'.join(f"{a:.6f} {b:.6f} {d:.6f}" for a, b, d in Q)
        data = raw[:start] + block.encode('latin1') + txt[pos:].encode('latin1')
    open(out, 'wb').write(data)


# ---------------- volume of a closed mesh (only for the S suggestion) ----------------
def read_ply_faces(fn):
    raw = open(fn, 'rb').read()
    end, fmt, elements = _ply_header(raw)
    if fmt != 'binary_little_endian':
        raise ValueError("volume suggestion needs a binary PLY")
    vname, nv, vprops = elements[0]
    vdt = np.dtype([(p[1], '<' + PLY_TYPES[p[0]]) for p in vprops])
    fname, nf, fprops = elements[1]
    cnt, idx = PLY_TYPES[fprops[0][1]], PLY_TYPES[fprops[0][2]]
    off = end + nv * vdt.itemsize
    fdt = np.dtype([('n', '<' + cnt), ('i', '<' + idx, 3)] +
                   [(p[1], '<' + PLY_TYPES[p[0]]) for p in fprops[1:]])
    F = np.frombuffer(raw, dtype=fdt, count=nf, offset=off)
    if not (F['n'] == 3).all():
        raise ValueError("volume suggestion needs a triangle mesh")
    return F['i'].astype(np.int64)


def read_vtk_faces(fn):
    L = open(fn, 'rb').read().decode('latin1').split('\n')
    j = next(k for k, l in enumerate(L) if l.startswith('POLYGONS'))
    m = int(L[j].split()[1])
    return np.array([list(map(int, L[j + 1 + r].split()[1:4])) for r in range(m)])


def volume(fn):
    if fn.lower().endswith('.ply'):
        P, F = read_ply_points(fn), read_ply_faces(fn)
    else:
        P, F = read_vtk_points(fn), read_vtk_faces(fn)
    a, b, d = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return abs(np.einsum('ij,ij->i', a, np.cross(b, d)).sum() / 6.0)


def points(fn):
    return read_ply_points(fn) if fn.lower().endswith('.ply') else read_vtk_points(fn)


def main():
    global S
    if S is None:
        if not REFERENCE:
            raise SystemExit("Set S, or set REFERENCE (closed target mesh) to get a suggestion from volumes.")
        vf, vr = volume(MESH), volume(REFERENCE)
        print(f"Volumes: fossil {vf:.1f}, reference {vr:.1f} (mesh units^3) -> suggested S = {(vr / vf) ** (1 / 3):.4f}")
        print("Check it, then put it in S and run again.")
        return
    c = points(MESH).mean(0)
    print(f"Scaling by S = {S} about the centroid of the mesh {c.round(3)}")
    for fn in [MESH] + list(CURVES):
        root, ext = os.path.splitext(fn)
        out = f"{root}_S{S:g}{ext}"
        (scale_ply if ext.lower() == '.ply' else scale_vtk)(fn, out, S, c)
        P0, P1 = points(fn), points(out)
        print(f"  {os.path.basename(out)}: {len(P1)} points, extent {np.ptp(P0, 0).round(2)} -> {np.ptp(P1, 0).round(2)}")
    print("Done. Align the _S files with the rigid script (registrationandlinesets4.py).")


if __name__ == "__main__":
    main()

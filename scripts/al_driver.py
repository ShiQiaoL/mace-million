#!/usr/bin/env python
"""
On-the-fly active learning with a MACE teacher (VASP-MLFF-like, with MACE-MPA-0 as the whole-system teacher).
Element-general: elements, thresholds and ensemble are arguments; nothing is specific to water or silica.

Each cycle (state is saved after every cycle; a new job resumes cleanly from --work-dir):
  1. build the three-pass LAMMPS model of the current student (make_threepass_model.py)
  2. run one MD segment in LAMMPS with the student (inputs/in.al_segment.lmp), dumping frames every --dump-every steps
  3. spot-check: teacher and student on the dumped frames -> force RMSE, max per-atom error, pressure difference
  4. if any threshold is exceeded (or the segment stopped early): add checked frames (or masked 13-A clusters for
     systems larger than --max-train-atoms) plus one-time condensed-phase short-range repulsive pair anchors, and
     fine-tune the student against a fixed held-out validation set (no cross-cycle validation leakage)
  5. continue from segment_end.restart if the segment remained physical; if halted or max |dF| > --rollback-fmax,
     mandatorily rewind to the last clean pre-segment checkpoint and replay the segment with the repaired student.

    python scripts/al_driver.py --work-dir AL --teacher mpa0.model --student V1.model --train-xyz train_ref.extxyz \
        --data start.data --elems O H --temp 300 --press 1.01325 --segment-steps 10000 --cycles 5
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import islice, zip_longest

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)          # repository root (holds inputs/ and the mace_onegpu package)
EV_A3_TO_BAR = 1.602176634e6


def sh(cmd, log, env=None, timeout=None):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            fh.write(f"Timed out after {exc.timeout} seconds\n")
            return 124
        except FileNotFoundError as exc:
            fh.write(f"{exc}\n")
            return 127
    return r.returncode


def load_state(wd):
    p = os.path.join(wd, "state.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def save_state(wd, st):
    with open(os.path.join(wd, "state.json"), "w") as fh:
        json.dump(st, fh, indent=1)


def model_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fixed_split(frames, valid_fraction):
    """Deterministic cycle-0-locked fixed train/validation split created once at cycle 0 so held-out validation frames never leak into training across subsequent active-learning cycles."""
    if valid_fraction <= 0.0 or len(frames) < 4:
        return list(frames), []
    groups = {}
    for idx, at in enumerate(frames):
        ct = str(at.info.get("config_type", "Default"))
        groups.setdefault(ct, []).append(idx)
    train_idx, valid_idx = [], []
    if len(groups) == 1 and len(frames) >= 20:
        only = next(iter(groups.values()))
        n_blocks = min(12, max(2, len(only) // 16))
        for blk in np.array_split(only, n_blocks):
            blk = blk.tolist()
            n_val = max(1, int(round(valid_fraction * len(blk))))
            train_idx.extend(blk[:-n_val])
            valid_idx.extend(blk[-n_val:])
    else:
        for indices in groups.values():
            if len(indices) < 2:
                train_idx.extend(indices)
            else:
                n_val = max(1, int(round(valid_fraction * len(indices))))
                train_idx.extend(indices[:-n_val])
                valid_idx.extend(indices[-n_val:])
    return [frames[i] for i in sorted(train_idx)], [frames[i] for i in sorted(valid_idx)]


def build_short_range_anchors(base_frames, elems, teacher, max_atoms=4000, n_dist=7, inner_radius=6.0,
                              has_nodeweights=False, seed=0):
    """Element-general condensed-phase short-range repulsive pair anchors spanning core overlap up to just below
    the empirical minimum pair distance in `base_frames` for every unordered element pair in `elems`.
    Energy and stress weights are zeroed so short-range repulsive anchoring never perturbs bulk offsets or virials."""
    from ase.data import atomic_numbers, covalent_radii
    from ase.geometry import get_distances

    pframes = [at for at in base_frames if all(at.pbc) and 4 <= len(at) <= max_atoms]
    if not pframes:
        return []
    rng = np.random.default_rng(seed)
    sample = [pframes[i] for i in np.linspace(0, len(pframes) - 1, min(36, len(pframes)), dtype=int)]
    zs = sorted({atomic_numbers[e] for e in elems})
    pairs_z = [(za, zb) for idx, za in enumerate(zs) for zb in zs[idx:]]

    d_eq_min = {}
    for za, zb in pairs_z:
        dmin = np.inf
        for at in sample:
            ia = np.where(at.numbers == za)[0]
            ib = np.where(at.numbers == zb)[0]
            if not len(ia) or not len(ib):
                continue
            _, dmat = get_distances(at.positions[ia], at.positions[ib], cell=at.cell, pbc=True)
            if za == zb:
                np.fill_diagonal(dmat, np.inf)
            if dmat.size:
                dmin = min(dmin, float(dmat.min()))
        rcov_sum = float(covalent_radii[za] + covalent_radii[zb])
        d_eq_min[(za, zb)] = dmin if np.isfinite(dmin) else 1.15 * rcov_sum

    anchors = []
    mask_r = inner_radius if inner_radius > 0 else 6.0
    pool = sample[len(sample) // 2:] or sample
    for za, zb in pairs_z:
        rcov_sum = float(covalent_radii[za] + covalent_radii[zb])
        d_lo = max(0.55, 0.60 * rcov_sum) if (za == 1 or zb == 1) else max(0.78, 0.50 * rcov_sum)
        d_hi = min(0.95 * d_eq_min[(za, zb)], 1.25 * rcov_sum)
        if d_hi <= d_lo + 0.12:
            d_hi = max(d_lo + 0.25, 0.96 * d_eq_min[(za, zb)])
        grid = np.linspace(d_lo, d_hi, max(3, n_dist))
        for d_target in grid:
            for co_move in (False, True):
                at0 = pool[int(rng.integers(len(pool)))]
                ia = np.where(at0.numbers == za)[0]
                ib = np.where(at0.numbers == zb)[0]
                if not len(ia) or not len(ib) or (za == zb and len(ia) < 2):
                    continue
                vecs, dmat = get_distances(at0.positions[ia], at0.positions[ib], cell=at0.cell, pbc=True)
                if za == zb:
                    np.fill_diagonal(dmat, np.inf)
                hi_cand = max(float(d_target) + 0.4, min(4.5, d_eq_min[(za, zb)] + 1.4))
                cand = np.argwhere((dmat > float(d_target) + 0.02) & (dmat < hi_cand))
                if not len(cand):
                    cand = np.argwhere(np.isfinite(dmat))
                if not len(cand):
                    continue
                rng.shuffle(cand)
                best_at, best_clr = None, -1.0
                for ca, cb in cand[:24]:
                    i, j = int(ia[ca]), int(ib[cb])
                    rij = vecs[ca, cb]
                    d0 = float(np.linalg.norm(rij))
                    if d0 < 1e-6:
                        continue
                    u = rij / d0
                    shift = 0.5 * (d0 - float(d_target))
                    pos = at0.positions.copy()
                    vi, dv_i = get_distances(at0.positions[i:i + 1], at0.positions, cell=at0.cell, pbc=True)
                    vj, dv_j = get_distances(at0.positions[j:j + 1], at0.positions, cell=at0.cell, pbc=True)
                    vi, dv_i = vi[0], dv_i[0]
                    vj, dv_j = vj[0], dv_j[0]
                    pos[i] += shift * u
                    pos[j] -= shift * u
                    if co_move:
                        rcov_all = covalent_radii[at0.numbers]
                        for k in range(len(at0)):
                            if k == i or k == j:
                                continue
                            if dv_i[k] < 1.22 * (covalent_radii[at0.numbers[i]] + rcov_all[k]) and (
                                at0.numbers[k] == 1 or np.dot(vi[k], u) < 0.0
                            ):
                                pos[k] += (1.0 if at0.numbers[k] == 1 else 0.5) * shift * u
                            elif dv_j[k] < 1.22 * (covalent_radii[at0.numbers[j]] + rcov_all[k]) and (
                                at0.numbers[k] == 1 or np.dot(vj[k], u) > 0.0
                            ):
                                pos[k] -= (1.0 if at0.numbers[k] == 1 else 0.5) * shift * u
                    _, di_new = get_distances(pos[i:i + 1], pos, cell=at0.cell, pbc=True)
                    _, dj_new = get_distances(pos[j:j + 1], pos, cell=at0.cell, pbc=True)
                    others = [k for k in range(len(at0)) if k != i and k != j]
                    clr = float(min(di_new[0, others].min(), dj_new[0, others].min())) if others else 9.9
                    if clr > best_clr:
                        anc = at0.copy()
                        anc.positions = pos
                        anc.wrap()
                        best_at, best_clr = (anc, i, j), clr
                        if clr >= max(1.15, 0.85 * d_eq_min[(za, zb)]):
                            break
                if best_at is None:
                    continue
                anc, i, j = best_at
                anc.calc = teacher
                Et, Ft = float(anc.get_potential_energy()), anc.get_forces()
                anc.calc = None
                if not np.isfinite(Ft).all() or np.abs(Ft).max() > 2000.0:
                    continue
                anc.info = dict(
                    REF_energy=Et,
                    config_type="sr_anchor",
                    config_energy_weight=0.0,
                    config_stress_weight=0.0,
                    anchor_pair=f"{za}-{zb}",
                    anchor_d=float(d_target),
                    anchor_comove=bool(co_move),
                )
                for k in list(anc.arrays):
                    if k not in ("numbers", "positions"):
                        del anc.arrays[k]
                anc.arrays["REF_forces"] = Ft
                if has_nodeweights and mask_r > 0:
                    _, di_m = get_distances(anc.positions[i:i + 1], anc.positions, cell=anc.cell, pbc=True)
                    _, dj_m = get_distances(anc.positions[j:j + 1], anc.positions, cell=anc.cell, pbc=True)
                    anc.arrays["forces_weight"] = ((di_m[0] <= mask_r) | (dj_m[0] <= mask_r)).astype(float)
                anchors.append(anc)
    return anchors


class FrameCheckError(ValueError):
    """A trajectory, force rerun or direct frame evaluation cannot be paired and scored."""


def check_frames(frames, teacher, student, f_thr, skip_first=False):
    """Per frame: force RMSE, max per-atom |dF|, pressure difference (virial part), all student vs teacher."""
    rows, labelled = [], []
    for i, at in enumerate(frames):
        if not np.isfinite(at.positions).all():
            raise FrameCheckError(f"non-finite positions in frame {i}")
        a = at.copy()
        a.calc = teacher
        Ft, Et, St = a.get_forces(), a.get_potential_energy(), a.get_stress(voigt=False)
        b = at.copy()
        b.calc = student
        Fs, Ss = b.get_forces(), b.get_stress(voigt=False)
        if Ft.shape != (len(at), 3) or Fs.shape != Ft.shape:
            raise FrameCheckError(f"force count does not match the atoms in frame {i}")
        if not all(np.isfinite(x).all() for x in (Ft, Fs, Et, St, Ss)):
            raise FrameCheckError(f"non-finite force, energy or stress in frame {i}")
        if skip_first and i == 0:
            continue
        dF = np.linalg.norm(Fs - Ft, axis=1)
        dP = (-np.trace(Ss) + np.trace(St)) / 3 * EV_A3_TO_BAR
        rows.append(dict(f_rmse=float(np.sqrt(((Fs - Ft) ** 2).mean())), f_max=float(dF.max()), dP_bar=float(dP),
                         frac_over=float((dF > f_thr).mean()),
                         density=float(sum(at.get_masses()) / 0.602214076 / at.get_volume())))
        c = at.copy()
        c.calc = None
        c.info["REF_energy"] = Et
        c.arrays["REF_forces"] = Ft
        c.info["REF_stress"] = St.reshape(-1)
        c.info["config_type"] = "al"
        labelled.append(c)
    return rows, labelled


def dump_frames(path, columns):
    """Yield each complete frame as (timestep, sorted integer atom IDs, three-vector)."""
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.strip() != "ITEM: TIMESTEP":
                    raise ValueError("expected ITEM: TIMESTEP")
                step = int(fh.readline())
                if fh.readline().strip() != "ITEM: NUMBER OF ATOMS":
                    raise ValueError(f"missing atom count at step {step}")
                n = int(fh.readline())
                if n <= 0 or not fh.readline().startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(f"invalid atom count or box at step {step}")
                for _ in range(3):
                    bounds = [float(x) for x in fh.readline().split()]
                    if len(bounds) < 2 or not np.isfinite(bounds).all():
                        raise ValueError(f"invalid box bounds at step {step}")
                header = fh.readline().split()
                if header[:2] != ["ITEM:", "ATOMS"]:
                    raise ValueError(f"missing atom columns at step {step}")
                names = header[2:]
                extra = [name for name in ("x", "y", "z", "fx", "fy", "fz") if name in names and name not in columns]
                usecols = [names.index(name) for name in ("id", *columns, *extra)]
                data = np.loadtxt(islice(fh, n), usecols=usecols, comments=None, ndmin=2)
                if data.shape != (n, len(usecols)) or not np.isfinite(data).all():
                    raise ValueError(f"truncated or non-finite atom data at step {step}")
                ids = data[:, 0].astype(np.int64)
                if not np.array_equal(ids, data[:, 0]) or np.any(ids <= 0):
                    raise ValueError(f"invalid atom IDs at step {step}")
                values = data[:, 1:4]
                if np.any(ids[1:] <= ids[:-1]):
                    order = ids.argsort()
                    ids, values = ids[order], values[order]
                    if np.any(ids[1:] == ids[:-1]):
                        raise ValueError(f"duplicate atom IDs at step {step}")
                yield step, ids, values
    except (FileNotFoundError, ValueError) as exc:
        raise FrameCheckError(f"{path}: {exc}") from exc


def trajectory_index(path, start_step, segment_steps, every, halted):
    """Validate positions and identities even for the initial frame omitted from the error statistics."""
    steps, atom_ids = [], None
    for step, ids, _ in dump_frames(path, ("x", "y", "z")):
        if atom_ids is None:
            atom_ids = ids
        elif not np.array_equal(ids, atom_ids):
            raise FrameCheckError(f"trajectory atom IDs or count changed at step {step}")
        steps.append(step)
    expected = range(start_step + (-start_step % every), start_step + segment_steps + 1, every)
    if not steps or steps != list(expected[:len(steps)] if halted else expected):
        raise FrameCheckError(f"trajectory steps {steps} disagree with the dump schedule {expected}")
    return steps, atom_ids


def paired_dump_checks(trajectory_steps, atom_ids, teacher, student, density, f_thr, skip_first):
    """Compare each rerun with the source trajectory before using any frame for retraining."""
    teacher_path, pT = teacher
    student_path, pS = student
    fT = dump_frames(teacher_path, ("fx", "fy", "fz"))
    fS = dump_frames(student_path, ("fx", "fy", "fz"))
    rows, dFs = [], {}
    try:
        for k, (step, t, s) in enumerate(zip_longest(trajectory_steps, fT, fS)):
            if t is None or s is None or t[0] != step or s[0] != step:
                raise FrameCheckError(f"missing or mismatched rerun timestep at frame {k} (expected {step})")
            if not np.array_equal(t[1], atom_ids) or not np.array_equal(s[1], atom_ids):
                raise FrameCheckError(f"rerun atom IDs or count differ from trajectory at step {step}")
            if not np.isfinite([pT.get(step, np.nan), pS.get(step, np.nan)]).all():
                raise FrameCheckError(f"missing or non-finite rerun pressure at step {step}")
            if skip_first and k == 0:
                continue
            delta = s[2] - t[2]
            dF = np.linalg.norm(delta, axis=1)
            dFs[k] = dF
            rows.append(dict(frame=k, step=step, f_rmse=float(np.sqrt((delta ** 2).mean())), f_max=float(dF.max()),
                             frac_over=float((dF > f_thr).mean()), dP_bar=float(pS[step] - pT[step]),
                             density=density.get(step)))
    finally:
        fT.close()
        fS.close()
    return rows, dFs


def segment_gates(rows, halted, a):
    worst = {k: max((abs(r[k]) for r in rows), default=None) for k in ("f_rmse", "f_max", "frac_over", "dP_bar")}
    exceed_rollback = a.rollback_fmax > 0 and worst["f_max"] is not None and worst["f_max"] > a.rollback_fmax
    bad = (halted or not rows or exceed_rollback or worst["f_rmse"] > a.f_rmse_max
           or worst["frac_over"] > a.f_over_frac or worst["dP_bar"] > a.dp_max)
    rollback = halted or not rows or exceed_rollback
    return worst, bad, rollback


def read_rerun_pressure(log):
    """Thermo lines 'Step PotEng c_pv' of a rerun log -> {step: virial pressure (bar)}."""
    res, on = {}, False
    with open(log, errors="replace") as fh:
        for ln in fh:
            w = ln.split()
            if w[:3] == ["Step", "PotEng", "c_pv"]:
                on = True
                continue
            if on:
                try:
                    res[int(w[0])] = float(w[2])
                except (ValueError, IndexError):
                    on = False
    return res


def read_rerun_pe(log):
    """Thermo lines 'Step PotEng c_pv' of a rerun log -> {step: potential energy (eV)}."""
    res, on = {}, False
    with open(log, errors="replace") as fh:
        for ln in fh:
            w = ln.split()
            if w[:3] == ["Step", "PotEng", "c_pv"]:
                on = True
                continue
            if on:
                try:
                    res[int(w[0])] = float(w[1])
                except (ValueError, IndexError):
                    on = False
    return res


def read_thermo_density(log):
    """{step: density} from the thermo lines of an in.al_segment.lmp log."""
    res, col = {}, None
    with open(log, errors="replace") as fh:
        for ln in fh:
            w = ln.split()
            if w[:1] == ["Step"] and "Density" in w:
                col = w.index("Density")
                continue
            if col is not None:
                try:
                    res[int(w[0])] = float(w[col])
                except (ValueError, IndexError):
                    col = None
    return res


def lammps_check(cd, st, model, tag, a, env):
    """Forces + virial pressure of all frames of cd/segment.lammpstrj with one LAMMPS model (rerun)."""
    shutil.copy(os.path.join(KIT, "inputs", "in.al_check.lmp"), os.path.join(cd, "in.check.lmp"))
    check_env = dict(env, MACE3P_FMAX_ABORT="1e6")
    cmd = ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half", "-in", "in.check.lmp",
           "-log", f"log.check_{tag}", "-var", "start_file", st["start"], "-var", "restart", str(st["restart"]),
           "-var", "model_file", model, "-var", "elems", " ".join(a.elems), "-var", "traj", "segment.lammpstrj",
           "-var", "dump_every", str(a.dump_every), "-var", "tag", tag]
    with open(os.path.join(cd, f"check_{tag}.out"), "w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=cd, stdout=fh, stderr=subprocess.STDOUT, env=check_env, timeout=3600).returncode
        except subprocess.TimeoutExpired as exc:
            raise FrameCheckError(f"{tag} rerun timed out after {exc.timeout} seconds") from exc
    if rc != 0:
        raise FrameCheckError(f"LAMMPS rerun with {tag} failed (rc={rc}); see {cd}/check_{tag}.out")
    fd = os.path.join(cd, f"forces_{tag}.dump")
    return fd, read_rerun_pressure(os.path.join(cd, f"log.check_{tag}"))


def carve_clusters(at, dF, n_clusters, radius, max_atoms, bond_scale=1.2, closure=3, mode="worst", seed=0, inner=0.0):
    """Non-periodic clusters around the atoms with the largest student error, separated by >= 2*radius.
    Element-general: bonds cut by the sphere are completed with covalent radii (ase natural_cutoffs x bond_scale),
    so no dangling fragments; the cluster is recentred in a vacuum box (pbc False, no stress label).
    inner > 0: per-atom force weights (array forces_weight) = 1 within `inner` of the centre, 0 in the buffer beyond,
    so that only the inner region, whose labels match the bulk, enters the force loss (needs the patched MACE)."""
    from ase import Atoms
    from ase.data import covalent_radii
    from ase.geometry import find_mic

    order = np.argsort(-dF) if mode == "worst" else np.random.default_rng(seed).permutation(len(dF))
    centres, out = [], []
    rcov = covalent_radii[at.numbers] * bond_scale
    for c in order:
        if len(centres) >= n_clusters:
            break
        if any(np.linalg.norm(find_mic(at.positions[c] - at.positions[o], at.cell, True)[0]) < 2 * radius for o in centres):
            continue
        centres.append(c)
        d, dist = find_mic(at.positions - at.positions[c], at.cell, True)
        sel = dist < radius
        shell = np.where((dist >= radius) & (dist < radius + 2 * rcov.max() * closure))[0]
        for _ in range(closure):              # add atoms bonded to the current cluster (completes molecules / bonds)
            ins = np.where(sel)[0]
            cand = shell[~sel[shell]]
            if not len(cand):
                break
            rij = np.linalg.norm(d[cand][:, None, :] - d[ins][None, :, :], axis=2)
            bonded = (rij < rcov[cand][:, None] + rcov[ins][None, :]).any(1)
            if not bonded.any():
                break
            sel[cand[bonded]] = True
        if sel.sum() > max_atoms:
            continue
        cl = Atoms(numbers=at.numbers[sel], positions=d[sel], pbc=False)
        cl.center(vacuum=8.0)
        cl.info["carve_centre"] = int(c)
        cl.arrays["parent_index"] = np.where(sel)[0]          # index of every cluster atom in the periodic frame
        if inner > 0:
            cl.arrays["forces_weight"] = (dist[sel] <= inner).astype(float)
            cl.info["inner_radius"] = float(inner)
        out.append(cl)
    return out


def parse_args(argv=None):
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=("default", "jctc-hardened"), default="default",
                               help="jctc-hardened selects the published masked, anchored repair protocol")
    ap = argparse.ArgumentParser(parents=[preset_parser])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True, help="initial student .model (used only when starting fresh)")
    ap.add_argument("--train-xyz", required=True, help="initial training set (REF_energy/REF_forces/REF_stress)")
    ap.add_argument("--valid-file", default=None, help="optional external fixed validation .extxyz")
    ap.add_argument("--valid-fraction", type=float, default=0.1,
                    help="fraction of initial training frames held out in a fixed validation file created once at cycle 0")
    ap.add_argument("--fixed-val-split", type=int, choices=[0, 1], default=1,
                    help="1 locks validation at cycle 0 (default); 0 lets the trainer split each training set")
    ap.add_argument("--data", required=True, help="LAMMPS data file for the first segment")
    ap.add_argument("--elems", nargs="+", required=True, help="element names in LAMMPS type order")
    ap.add_argument("--e0s", required=True, help='e.g. "{1: -1.117, 8: -1.548}" (teacher E0s)')
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--press", type=float, default=1.01325, help="bar; <= 0 -> NVT")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--segment-steps", type=int, default=10000)
    ap.add_argument("--dump-every", type=int, default=2000)
    ap.add_argument("--cycles", type=int, default=5, help="target number of accepted MD segments")
    ap.add_argument("--time-budget-s", type=float, default=6000)
    ap.add_argument("--f-rmse-max", type=float, default=0.035, help="eV/A")
    ap.add_argument("--f-max-max", type=float, default=0.5, help="eV/A, per-atom |dF| threshold")
    ap.add_argument("--f-over-frac", type=float, default=1e-5,
                    help="retrain if more than this fraction of atoms exceeds --f-max-max (small frames: any atom)")
    ap.add_argument("--dp-max", type=float, default=300.0, help="bar")
    ap.add_argument("--rollback-fmax", type=float, default=5.0,
                    help="eV/A; > 0: if max |dF| exceeds this corruption threshold (or LAMMPS halts), rewind to the last "
                         "clean checkpoint before continuing MD with the repaired student (0 disables force-based rollback)")
    ap.add_argument("--max-rollbacks", type=int, default=3,
                    help="stop when this many mandatory rollbacks have occurred; "
                         "0 permits clean segments but stops at the first required rewind without replay")
    ap.add_argument("--max-train-atoms", type=int, default=4000,
                    help="larger frames: LAMMPS spot-check (--teacher-lmp) + carved clusters as training data")
    ap.add_argument("--teacher-lmp", default=None, help="LAMMPS ML-IAP .pt of the teacher (needed for large frames)")
    ap.add_argument("--carve-radius", type=float, default=13.0,
                    help="A; cluster radius before bond completion. 13 A reproduces the bulk correction at the centre atoms to "
                         "~1%% for water and SiO2 (7 A: ~50%%; CuNi is converged at 7 A); see experimental/validate_clusters.py")
    ap.add_argument("--carve-per-frame", type=int, default=4)
    ap.add_argument("--carve-frames", type=int, default=2)
    ap.add_argument("--carve-mode", choices=["worst", "random"], default="worst",
                    help="centres of the carved clusters: atoms with the largest error, or random atoms (control)")
    ap.add_argument("--al-weight", type=float, default=1.0, help="loss weight of frames acquired on the fly (config_type al / al_cluster)")
    ap.add_argument("--short-range-anchors", type=int, default=1, choices=[0, 1],
                    help="1 (default): include teacher-labelled condensed-phase short-range repulsive pair anchors during fine-tuning")
    ap.add_argument("--sr-anchor-weight", type=float, default=5.0, help="config_type weight for short-range repulsive anchors")
    ap.add_argument("--sr-anchor-points", type=int, default=7, help="number of short-range distance grid points per element pair")
    ap.add_argument("--md-seed", type=int, default=20260926, help="LAMMPS velocity seed of cycle c is md_seed + c (paired arms: same seed)")
    ap.add_argument("--train-seed", type=int, default=1, help="base random seed for training minibatches, anchors, and random carving")
    ap.add_argument("--inner-radius", type=float, default=0.0,
                    help="A; > 0: only cluster atoms within this distance of the centre carry force loss (array forces_weight; "
                         "needs the per-atom-weight MACE patch on PYTHONPATH). 0 = all cluster atoms")
    ap.add_argument("--cluster-energy", type=int, default=1, choices=[0, 1],
                    help="1: cluster total energies enter the energy loss; 0: config_energy_weight = 0 for clusters")
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--finetune-lr", type=float, default=0.002, help="learning rate for student fine-tuning")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--teacher-chunk", type=int, default=16384, help="receiver chunk for the teacher LAMMPS check (memory)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the student in LAMMPS (MACE3P_COMPILE=1)")
    ap.add_argument("--segment-timeout", type=int, default=3600, help="s, wall-time limit of one MD segment")
    preset, _ = preset_parser.parse_known_args(argv)
    if preset.preset == "jctc-hardened":
        ap.set_defaults(inner_radius=6.0, cluster_energy=0, al_weight=10.0, short_range_anchors=1,
                        sr_anchor_points=8, sr_anchor_weight=5.0, finetune_lr=0.002, rollback_fmax=5.0,
                        fixed_val_split=1)
    args = ap.parse_args(argv)
    if args.max_rollbacks < 0:
        ap.error("--max-rollbacks must be nonnegative")
    return args


def main(argv=None):
    a = parse_args(argv)
    if a.teacher_lmp:
        a.teacher_lmp = os.path.abspath(a.teacher_lmp)
    t_job = time.time()

    import mace.modules.loss as _ml
    has_nodeweights = hasattr(_ml, "node_forces_weight")
    if a.inner_radius > 0 and not has_nodeweights:
        sys.exit("--inner-radius needs the per-atom force-weight MACE patch (patches/patch_mace_nodeweights.py)")

    from ase.data import atomic_numbers
    from ase.io import read, write
    from mace.calculators import MACECalculator

    wd = os.path.abspath(a.work_dir)
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, "driver.log")
    st = load_state(wd)
    if st is None:
        shutil.copy(a.student, os.path.join(wd, "student_c0.model"))
        raw_frames = read(a.train_xyz, index=":")
        valid_path = os.path.join(wd, "valid_fixed.extxyz")
        train0_path = os.path.join(wd, "train_c0.extxyz")
        if a.valid_file:
            shutil.copy(a.valid_file, valid_path)
            shutil.copy(a.train_xyz, train0_path)
        elif a.fixed_val_split:
            tr_frames, val_frames = prepare_fixed_split(raw_frames, a.valid_fraction)
            write(train0_path, tr_frames)
            if val_frames:
                write(valid_path, val_frames)
            elif a.preset == "jctc-hardened":
                raise ValueError("fixed validation needs held-out frames: supply --valid-file or a larger initial training set")
            else:
                valid_path = None
        else:
            shutil.copy(a.train_xyz, train0_path)
            valid_path = None
        st = dict(cycle=0, accepted=0, rollbacks=0, anchored=False, stop_reason=None,
                  student=os.path.join(wd, "student_c0.model"),
                  train=train0_path, valid=valid_path,
                  start=os.path.abspath(a.data), restart=0, history=[])
        save_state(wd, st)
    else:
        st.setdefault("accepted", sum(1 for h in st.get("history", []) if not h.get("rolled_back", False)))
        st.setdefault("rollbacks", sum(1 for h in st.get("history", []) if h.get("rolled_back", False)))
        st.setdefault("anchored", os.path.exists(os.path.join(wd, "sr_anchors.extxyz")))
        if "valid" not in st:
            vp = os.path.join(wd, "valid_fixed.extxyz")
            if a.valid_file:
                shutil.copy(a.valid_file, vp)
                st["valid"] = vp
            elif a.fixed_val_split:
                tr_f, val_f = prepare_fixed_split(read(st["train"], index=":"), a.valid_fraction)
                if val_f:
                    tr_split = os.path.join(wd, f"train_split_c{st['cycle']}.extxyz")
                    write(tr_split, tr_f)
                    write(vp, val_f)
                    st["train"], st["valid"] = tr_split, vp
                else:
                    st["valid"] = None
            else:
                st["valid"] = None
            save_state(wd, st)

    teacher = MACECalculator(model_paths=a.teacher, device="cuda", default_dtype="float64")
    zmap = {i + 1: atomic_numbers[e] for i, e in enumerate(a.elems)}
    # mkl-service sets MKL_THREADING_LAYER=INTEL when numpy is imported here; children would then clash with libgomp
    # the repository root goes on PYTHONPATH so that the Python embedded in LAMMPS can unpickle mace_onegpu.* classes
    pypath = os.pathsep.join(p for p in (KIT, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, MACE3P_GRAPHS="1", MACE3P_CHUNK=str(a.chunk), MKL_THREADING_LAYER="GNU",
               MACE3P_COMPILE="1" if a.compile else "0", PYTHONPATH=pypath)

    rollback_limit = max(1, a.max_rollbacks)  # A rejected segment always rewinds, even with zero replays allowed.
    while (st["accepted"] < a.cycles and st["rollbacks"] < rollback_limit
           and time.time() - t_job < a.time_budget_s):
        c = st["cycle"]
        cd = os.path.join(wd, f"cycle{c:03d}")
        if os.path.exists(cd):
            # Preserve an interrupted attempt. Its restart, dumps and model must never be reused as new output.
            attempt = 1
            while os.path.exists(f"{cd}.interrupted-{attempt}"):
                attempt += 1
            os.rename(cd, f"{cd}.interrupted-{attempt}")
        os.mkdir(cd)
        t0 = time.time()
        # 1. LAMMPS model of the current student
        tp = os.path.join(cd, "student.threepass-mliap_lammps.pt")
        rc_b = sh([sys.executable, os.path.join(HERE, "make_threepass_model.py"), st["student"], "--out", tp, "--spline-grid", "512"],
                  log, env=env)
        if rc_b != 0 or not os.path.exists(tp):
            raise RuntimeError(f"building the LAMMPS model of {st['student']} failed (rc={rc_b}); see {log}")
        # 2. MD segment
        shutil.copy(os.path.join(KIT, "inputs", "in.al_segment.lmp"), os.path.join(cd, "in.lmp"))
        cmd = ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half", "-in", "in.lmp",
               "-log", "log.lammps", "-var", "start_file", st["start"], "-var", "restart", str(st["restart"]),
               "-var", "model_file", tp, "-var", "elems", " ".join(a.elems), "-var", "steps", str(a.segment_steps),
               "-var", "dt", str(a.dt), "-var", "temp", str(a.temp), "-var", "press", str(a.press),
               "-var", "dump_every", str(a.dump_every), "-var", "seed", str(a.md_seed + c)]
        with open(os.path.join(cd, "lammps.out"), "w") as fh:
            try:
                rc = subprocess.run(cmd, cwd=cd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=a.segment_timeout).returncode
            except subprocess.TimeoutExpired as exc:
                fh.write(f"MD segment timed out after {exc.timeout} seconds\n")
                rc = 124
        with open(os.path.join(cd, "lammps.out"), errors="replace") as fh:
            out = fh.read()
        restart_path = os.path.join(cd, "segment_end.restart")
        missing_restart = not os.path.isfile(restart_path)
        halted = ("Fix halt" in out) or ("guard" in out) or rc != 0 or missing_restart
        # 3. spot-check dumped frames (the segment's first frame = previous end is skipped)
        tr = os.path.join(cd, "segment.lammpstrj")
        nat, large, frame_error = 0, False, None
        rows, labelled = [], []
        try:
            dens = read_thermo_density(os.path.join(cd, "log.lammps"))
            start_step = next(iter(dens), st["accepted"] * a.segment_steps)
            steps, atom_ids = trajectory_index(tr, start_step, a.segment_steps, a.dump_every, halted)
            skip_first = len(steps) > 1 and steps[0] == start_step
            nat = len(atom_ids)
            large = nat > a.max_train_atoms
            if large:
                if not a.teacher_lmp:
                    raise RuntimeError(f"frames of {nat} atoms > --max-train-atoms: --teacher-lmp is required")
                t_dump = lammps_check(cd, st, a.teacher_lmp, "teacher", a,
                                      dict(env, MACE3P_CHUNK=str(a.teacher_chunk), MACE3P_COMPILE="0"))
                s_dump = lammps_check(cd, st, tp, "student", a, env)
                rows, dFs = paired_dump_checks(steps, atom_ids, t_dump, s_dump, dens, a.f_max_max, skip_first)
                for path, _ in (t_dump, s_dump):
                    os.remove(path)  # keep the source trajectory; each force dump is ~80 MB per million-atom frame
            else:
                frames = read(tr, format="lammps-dump-text", index=":")
                if len(frames) != len(steps):
                    raise FrameCheckError("ASE frame count differs from the trajectory")
                for at in frames:
                    at.numbers = np.array([zmap[int(z)] for z in at.numbers])
                    at.pbc = True
                student = MACECalculator(model_paths=st["student"], device="cuda", default_dtype="float64")
                rows, labelled = check_frames(frames, teacher, student, a.f_max_max, skip_first)
                del student
        except (FrameCheckError, FileNotFoundError) as exc:
            frame_error = str(exc)
            halted = True
        worst, bad, rollback = segment_gates(rows, halted, a)
        dl = [r["density"] for r in rows if r.get("density") is not None]
        rec = dict(cycle=c, accepted_before=st["accepted"], start_used=st["start"], restart_used=st["restart"],
                   rc=rc, halted=halted, bad=bad, frame_mismatch_or_nonfinite=frame_error is not None,
                   n_atoms=nat, large=large, n_checked=len(rows), worst=worst, checks=rows,
                   density_mean=float(np.mean(dl)) if dl else None)
        if frame_error is not None:
            rec["frame_error"] = frame_error
        if bad and large and rows:
            for r in sorted(rows, key=lambda r: -r["f_max"])[:a.carve_frames]:
                at = read(tr, format="lammps-dump-text", index=r["frame"])
                at.numbers = np.array([zmap[int(z)] for z in at.numbers])
                at.pbc = True
                carve_seed = 10000 * a.train_seed + 1000 * c + r["frame"]
                for cl in carve_clusters(at, dFs[r["frame"]], a.carve_per_frame, a.carve_radius, a.max_train_atoms,
                                         mode=a.carve_mode, seed=carve_seed, inner=a.inner_radius):
                    cl.calc = teacher
                    E, F = cl.get_potential_energy(), cl.get_forces()
                    cl.calc = None
                    cl.info.update(REF_energy=E, config_type="al_cluster")
                    if not a.cluster_energy:      # vacuum cluster total energy is not a bulk label: keep out of loss
                        cl.info["config_energy_weight"] = 0.0
                    cl.arrays["REF_forces"] = F
                    labelled.append(cl)
            rec["carved"] = [len(x) for x in labelled]
        # 4. retrain if needed
        new_student = st["student"]
        retrained = False
        if bad and labelled:
            base_train = read(st["train"], index=":")
            n_anchors = 0
            if a.short_range_anchors and not st.get("anchored", False):
                anc_path = os.path.join(wd, "sr_anchors.extxyz")
                if os.path.exists(anc_path):
                    anchors = read(anc_path, index=":")
                else:
                    anchors = build_short_range_anchors(
                        base_train, a.elems, teacher, max_atoms=a.max_train_atoms,
                        n_dist=a.sr_anchor_points, inner_radius=a.inner_radius,
                        has_nodeweights=has_nodeweights, seed=1000 * a.train_seed
                    )
                    if anchors:
                        write(anc_path, anchors)
                base_train = base_train + anchors
                n_anchors = len(anchors)
            add = [f for f in labelled if len(f) <= a.max_train_atoms]
            train_new = os.path.join(wd, f"train_c{c + 1}.extxyz")
            write(train_new, base_train + add)
            fdir = os.path.join(cd, "finetune")
            val_args = ["--valid_file", st["valid"]] if st.get("valid") else ["--valid_fraction", str(a.valid_fraction)]
            ct_weights = {"Default": 1.0, "al": a.al_weight, "al_cluster": a.al_weight}
            if a.short_range_anchors:
                ct_weights["sr_anchor"] = a.sr_anchor_weight
            wt_args = ["--config_type_weights", json.dumps(ct_weights)] if (a.al_weight != 1.0 or a.short_range_anchors) else []
            rc_t = sh(["mace_run_train", "--name", f"student_c{c + 1}", "--work_dir", fdir, "--foundation_model", st["student"],
                       "--multiheads_finetuning", "False", "--train_file", train_new] + val_args +
                      ["--E0s", a.e0s, "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
                       "--loss", "stress", "--energy_weight", "1", "--forces_weight", "100", "--stress_weight", "10000",
                       "--batch_size", "4", "--valid_batch_size", "4", "--max_num_epochs", str(a.finetune_epochs),
                       "--lr", str(a.finetune_lr), "--scheduler_patience", "5",
                       "--ema", "--ema_decay", "0.99", "--default_dtype", "float32", "--device", "cuda",
                       "--seed", str(100 * a.train_seed + c + 1)] + wt_args, log, env=env, timeout=3000)
            cand = os.path.join(fdir, f"student_c{c + 1}.model")
            train_error = None
            if rc_t != 0:
                train_error = f"trainer exited {rc_t}"
            else:
                try:
                    if not os.path.isfile(cand) or os.path.getsize(cand) == 0:
                        train_error = f"missing or empty candidate: {cand}"
                    else:
                        retrained = model_sha256(cand) != model_sha256(st["student"])
                except OSError as exc:
                    train_error = f"could not verify model content: {exc}"
            if retrained:
                new_student = os.path.join(wd, f"student_c{c + 1}.model")
                shutil.copy(cand, new_student)
                st["train"] = train_new
                st["anchored"] = st.get("anchored", False) or n_anchors > 0
            rec.update(added_frames=len(add), sr_anchors_added=n_anchors, retrained=retrained,
                       train_rc=rc_t, train_failed=train_error is not None)
            if train_error:
                rec["train_error"] = train_error
        # 5. next start point: advance if physical, or mandatorily rewind if halted or |dF|_max exceeded --rollback-fmax
        if not rollback:
            st["start"], st["restart"] = restart_path, 1
            st["accepted"] = st.get("accepted", 0) + 1
            rec["rolled_back"] = False
        else:
            st["rollbacks"] = st.get("rollbacks", 0) + 1
            rec["rolled_back"] = True
            rec["rewind_to"] = st["start"]
            if frame_error is not None:
                rec["rollback_reason"] = "frame_mismatch_or_nonfinite"
            elif missing_restart:
                rec["rollback_reason"] = "missing_end_restart"
            elif halted or not rows:
                rec["rollback_reason"] = "halted" if halted else "no_checked_frames"
            else:
                rec["rollback_reason"] = f"f_max={worst['f_max']:.2f}>{a.rollback_fmax:.2f}"
            if not retrained:
                rec["stop_reason"] = "rollback_without_model_update"
            elif st["rollbacks"] >= rollback_limit:
                rec["stop_reason"] = "max_rollbacks_exceeded"
        rec["t_cycle_s"] = time.time() - t0
        st["student"] = new_student
        st["history"].append(rec)
        st["cycle"] = c + 1
        st["stop_reason"] = rec.get("stop_reason")
        save_state(wd, st)
        if st["stop_reason"] == "rollback_without_model_update":
            cause = rec.get("frame_error", rec["rollback_reason"])
            raise RuntimeError(f"rollback_without_model_update in cycle {c}: {cause}; "
                               f"train_rc={rec.get('train_rc')}; previous restart preserved in {wd}/state.json")
        print(f"cycle {c} (acc {st['accepted']}/{a.cycles}): rc={rc} halted={halted} rollback={rec['rolled_back']} "
              f"checked={len(rows)} worst={worst} density={rec['density_mean']} "
              f"retrained={rec.get('retrained', False)} added={rec.get('added_frames', 0)} ({rec['t_cycle_s']:.0f}s)", flush=True)
    if st["rollbacks"] >= rollback_limit and not st.get("stop_reason"):
        # Older state files stored the reason only in history, or silently exhausted the budget.
        last_cycle = st["history"][-1] if st["history"] else {}
        st["stop_reason"] = last_cycle.get("stop_reason") or "max_rollbacks_exceeded"
        save_state(wd, st)
    print(f"done: {st['cycle']} cycles ({st['accepted']} accepted, {st['rollbacks']} rollbacks), "
          f"student = {st['student']}, stop_reason = {st.get('stop_reason')}", flush=True)


if __name__ == "__main__":
    main()

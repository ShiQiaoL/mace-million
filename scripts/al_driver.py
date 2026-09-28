#!/usr/bin/env python
"""
On-the-fly active learning with a MACE teacher (VASP-MLFF-like, but the reference is MACE-MPA-0 instead of DFT).
Element-general: elements, thresholds and ensemble are arguments; nothing is specific to water.

Each cycle (state is saved after every cycle, a new job resumes from --work-dir):
  1. build the three-pass LAMMPS model of the current student (make_threepass_model.py)
  2. run one MD segment in LAMMPS with the student (inputs/in.al_segment.lmp), dumping frames every --dump-every steps
  3. spot-check: teacher (fp64) and student on the dumped frames -> force RMSE, max per-atom error, pressure difference
  4. if any threshold is exceeded (or the segment stopped early): add the checked frames, labelled by the teacher,
     to the training set and fine-tune the student from its current weights (mace_run_train --foundation_model)
  5. continue from the segment's restart file (or from the last good restart if the segment failed)
Frames larger than --max-train-atoms are not added whole (cluster carving for very large systems: TODO v2).

    python scripts/al_driver.py --work-dir AL --teacher mpa0.model --student V1.model --train-xyz train_ref.extxyz \
        --data start.data --elems O H --temp 300 --press 1.01325 --segment-steps 10000 --cycles 5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)          # repository root (holds inputs/ and the mace_onegpu package)
EV_A3_TO_BAR = 1.602176634e6


def sh(cmd, log, env=None, timeout=None):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return r.returncode


def load_state(wd):
    p = os.path.join(wd, "state.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save_state(wd, st):
    json.dump(st, open(os.path.join(wd, "state.json"), "w"), indent=1)


def check_frames(frames, teacher, student, f_thr):
    """Per frame: force RMSE, max per-atom |dF|, pressure difference (virial part), all student vs teacher."""
    from ase.calculators.singlepoint import SinglePointCalculator

    rows, labelled = [], []
    for at in frames:
        a = at.copy()
        a.calc = teacher
        Ft, Et, St = a.get_forces(), a.get_potential_energy(), a.get_stress(voigt=False)
        b = at.copy()
        b.calc = student
        Fs, Ss = b.get_forces(), b.get_stress(voigt=False)
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


def read_force_dump(path):
    """LAMMPS 'id fx fy fz' dump (sorted by id) -> list of (timestep, forces [N, 3])."""
    out = []
    with open(path, errors="replace") as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("ITEM: TIMESTEP"):
        step = int(lines[i + 1])
        n = int(lines[i + 3])
        body = lines[i + 9:i + 9 + n]
        out.append((step, np.array(" ".join(body).split(), dtype=np.float64).reshape(n, 4)[:, 1:]))
        i += 9 + n
    return out


def read_rerun_pressure(log):
    """Thermo lines 'Step PotEng c_pv' of a rerun log -> {step: virial pressure (bar)}."""
    res, on = {}, False
    for ln in open(log, errors="replace"):
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


def read_thermo_density(log):
    """{step: density} from the thermo lines of an in.al_segment.lmp log."""
    res, col = {}, None
    for ln in open(log, errors="replace"):
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
    cmd = ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half", "-in", "in.check.lmp",
           "-log", f"log.check_{tag}", "-var", "start_file", st["start"], "-var", "restart", str(st["restart"]),
           "-var", "model_file", model, "-var", "elems", " ".join(a.elems), "-var", "traj", "segment.lammpstrj",
           "-var", "dump_every", str(a.dump_every), "-var", "tag", tag]
    with open(os.path.join(cd, f"check_{tag}.out"), "w") as fh:
        rc = subprocess.run(cmd, cwd=cd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=3600).returncode
    if rc != 0:
        raise RuntimeError(f"LAMMPS check with {tag} failed (rc={rc}); see {cd}/check_{tag}.out")
    fd = os.path.join(cd, f"forces_{tag}.dump")
    out = read_force_dump(fd), read_rerun_pressure(os.path.join(cd, f"log.check_{tag}"))
    os.remove(fd)                         # ~80 MB per 10^6-atom frame; the trajectory itself is kept
    return out


def carve_clusters(at, dF, n_clusters, radius, max_atoms, bond_scale=1.2, closure=3):
    """Non-periodic clusters around the atoms with the largest student error, separated by >= 2*radius.
    Element-general: bonds cut by the sphere are completed with covalent radii (ase natural_cutoffs x bond_scale),
    so no dangling fragments; the cluster is recentred in a vacuum box (pbc False, no stress label)."""
    from ase import Atoms
    from ase.data import covalent_radii
    from ase.geometry import find_mic

    order = np.argsort(-dF)
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
        out.append(cl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True, help="initial student .model (used only when starting fresh)")
    ap.add_argument("--train-xyz", required=True, help="initial training set (REF_energy/REF_forces/REF_stress)")
    ap.add_argument("--data", required=True, help="LAMMPS data file for the first segment")
    ap.add_argument("--elems", nargs="+", required=True, help="element names in LAMMPS type order")
    ap.add_argument("--e0s", required=True, help='e.g. "{1: -1.117, 8: -1.548}" (teacher E0s)')
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--press", type=float, default=1.01325, help="bar; <= 0 -> NVT")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--segment-steps", type=int, default=10000)
    ap.add_argument("--dump-every", type=int, default=2000)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--time-budget-s", type=float, default=6000)
    ap.add_argument("--f-rmse-max", type=float, default=0.035, help="eV/A")
    ap.add_argument("--f-max-max", type=float, default=0.5, help="eV/A, per-atom |dF| threshold")
    ap.add_argument("--f-over-frac", type=float, default=1e-5,
                    help="retrain if more than this fraction of atoms exceeds --f-max-max (small frames: any atom)")
    ap.add_argument("--dp-max", type=float, default=300.0, help="bar")
    ap.add_argument("--max-train-atoms", type=int, default=4000,
                    help="larger frames: LAMMPS spot-check (--teacher-lmp) + carved clusters as training data")
    ap.add_argument("--teacher-lmp", default=None, help="LAMMPS ML-IAP .pt of the teacher (needed for large frames)")
    ap.add_argument("--carve-radius", type=float, default=7.0, help="A; cluster radius before bond completion")
    ap.add_argument("--carve-per-frame", type=int, default=4)
    ap.add_argument("--carve-frames", type=int, default=2)
    ap.add_argument("--al-weight", type=float, default=1.0, help="loss weight of frames acquired on the fly (config_type al / al_cluster)")
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--teacher-chunk", type=int, default=16384, help="receiver chunk for the teacher LAMMPS check (memory)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the student in LAMMPS (MACE3P_COMPILE=1)")
    ap.add_argument("--segment-timeout", type=int, default=3600, help="s, wall-time limit of one MD segment")
    a = ap.parse_args()
    t_job = time.time()
    wd = os.path.abspath(a.work_dir)
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, "driver.log")
    st = load_state(wd)
    if st is None:
        shutil.copy(a.student, os.path.join(wd, "student_c0.model"))
        shutil.copy(a.train_xyz, os.path.join(wd, "train_c0.extxyz"))
        st = dict(cycle=0, student=os.path.join(wd, "student_c0.model"), train=os.path.join(wd, "train_c0.extxyz"),
                  start=os.path.abspath(a.data), restart=0, history=[])
        save_state(wd, st)
    from ase.data import atomic_numbers
    from ase.io import read, write
    from mace.calculators import MACECalculator

    teacher = MACECalculator(model_paths=a.teacher, device="cuda", default_dtype="float64")
    zmap = {i + 1: atomic_numbers[e] for i, e in enumerate(a.elems)}
    # mkl-service sets MKL_THREADING_LAYER=INTEL when numpy is imported here; children would then clash with libgomp
    # the repository root goes on PYTHONPATH so that the Python embedded in LAMMPS can unpickle mace_onegpu.* classes
    pypath = os.pathsep.join(p for p in (KIT, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, MACE3P_GRAPHS="1", MACE3P_CHUNK=str(a.chunk), MKL_THREADING_LAYER="GNU",
               MACE3P_COMPILE="1" if a.compile else "0", PYTHONPATH=pypath)
    while st["cycle"] < a.cycles and time.time() - t_job < a.time_budget_s:
        c = st["cycle"]
        cd = os.path.join(wd, f"cycle{c:03d}")
        os.makedirs(cd, exist_ok=True)
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
               "-var", "dump_every", str(a.dump_every), "-var", "seed", str(20260926 + c)]
        with open(os.path.join(cd, "lammps.out"), "w") as fh:
            rc = subprocess.run(cmd, cwd=cd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=a.segment_timeout).returncode
        out = open(os.path.join(cd, "lammps.out"), errors="replace").read()
        halted = ("Fix halt" in out) or ("guard" in out) or rc != 0
        # 3. spot-check dumped frames (the segment's first frame = previous end is skipped)
        tr = os.path.join(cd, "segment.lammpstrj")
        nat = 0
        if os.path.exists(tr):
            with open(tr, errors="replace") as fh:
                head = [fh.readline() for _ in range(4)]
            nat = int(head[3]) if head[2].startswith("ITEM: NUMBER") else 0
        large = nat > a.max_train_atoms
        rows, labelled = [], []
        if large:
            # v2, large frames: teacher and student forces/pressure in LAMMPS (rerun of the dump), exact and fast;
            # training data = clusters carved around the atoms with the largest errors, labelled by the teacher
            if not a.teacher_lmp:
                raise RuntimeError(f"frames of {nat} atoms > --max-train-atoms: --teacher-lmp is required")
            fT, pT = lammps_check(cd, st, a.teacher_lmp, "teacher", a,
                                  dict(env, MACE3P_CHUNK=str(a.teacher_chunk), MACE3P_COMPILE="0"))
            fS, pS = lammps_check(cd, st, tp, "student", a, env)
            dens = read_thermo_density(os.path.join(cd, "log.lammps"))
            dFs = {}
            for k, ((s, Ft), (_, Fs)) in enumerate(zip(fT, fS)):
                if (k == 0 and len(fT) > 1) or not (np.isfinite(Ft).all() and np.isfinite(Fs).all()):
                    continue
                dF = np.linalg.norm(Fs - Ft, axis=1)
                dFs[k] = dF
                rows.append(dict(frame=k, step=s, f_rmse=float(np.sqrt(((Fs - Ft) ** 2).mean())), f_max=float(dF.max()),
                                 frac_over=float((dF > a.f_max_max).mean()), dP_bar=float(pS[s] - pT[s]), density=dens.get(s)))
        else:
            frames = []
            if os.path.exists(tr):
                for at in read(tr, format="lammps-dump-text", index=":"):
                    at.numbers = np.array([zmap[int(z)] for z in at.numbers])
                    at.pbc = True
                    if np.isfinite(at.positions).all():
                        frames.append(at)
            frames = frames[1:] if len(frames) > 1 else frames
            student = MACECalculator(model_paths=st["student"], device="cuda", default_dtype="float64")
            rows, labelled = check_frames(frames, teacher, student, a.f_max_max) if frames else ([], [])
            del student
        worst = {k: max((abs(r[k]) for r in rows), default=None) for k in ("f_rmse", "f_max", "frac_over", "dP_bar")}
        # per-atom criterion as a fraction: the maximum over 10^6 atoms is always larger than over 10^3 atoms
        bad = halted or not rows or worst["f_rmse"] > a.f_rmse_max or worst["frac_over"] > a.f_over_frac \
            or worst["dP_bar"] > a.dp_max
        dl = [r["density"] for r in rows if r.get("density") is not None]
        rec = dict(cycle=c, rc=rc, halted=halted, n_atoms=nat, large=large, n_checked=len(rows), worst=worst, checks=rows,
                   density_mean=float(np.mean(dl)) if dl else None)
        if bad and large and rows:
            for r in sorted(rows, key=lambda r: -r["f_max"])[:a.carve_frames]:
                at = read(tr, format="lammps-dump-text", index=r["frame"])
                at.numbers = np.array([zmap[int(z)] for z in at.numbers])
                at.pbc = True
                for cl in carve_clusters(at, dFs[r["frame"]], a.carve_per_frame, a.carve_radius, a.max_train_atoms):
                    cl.calc = teacher
                    E, F = cl.get_potential_energy(), cl.get_forces()
                    cl.calc = None
                    cl.info.update(REF_energy=E, config_type="al_cluster")
                    cl.arrays["REF_forces"] = F
                    labelled.append(cl)
            rec["carved"] = [len(x) for x in labelled]
        # 4. retrain if needed
        new_student = st["student"]
        if bad and labelled:
            add = [f for f in labelled if len(f) <= a.max_train_atoms]
            train_new = os.path.join(wd, f"train_c{c + 1}.extxyz")
            write(train_new, read(st["train"], index=":") + add)
            fdir = os.path.join(cd, "finetune")
            rc_t = sh(["mace_run_train", "--name", f"student_c{c + 1}", "--work_dir", fdir, "--foundation_model", st["student"],
                       "--multiheads_finetuning", "False", "--train_file", train_new, "--valid_fraction", "0.1",
                       "--E0s", a.e0s, "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
                       "--loss", "stress", "--energy_weight", "1", "--forces_weight", "100", "--stress_weight", "10000",
                       "--batch_size", "4", "--valid_batch_size", "4", "--max_num_epochs", str(a.finetune_epochs),
                       "--lr", "0.005", "--ema", "--ema_decay", "0.99", "--default_dtype", "float32", "--device", "cuda",
                       "--seed", str(c + 1)] + (["--config_type_weights", json.dumps({"Default": 1.0, "al": a.al_weight, "al_cluster": a.al_weight})] if a.al_weight != 1.0 else []), log, env=env, timeout=3000)
            cand = os.path.join(fdir, f"student_c{c + 1}.model")
            if rc_t == 0 and os.path.exists(cand):
                new_student = os.path.join(wd, f"student_c{c + 1}.model")
                shutil.copy(cand, new_student)
                st["train"] = train_new
            rec.update(added_frames=len(add), retrained=new_student != st["student"], train_rc=rc_t)
        # 5. next start point: continue from this segment unless it ran away
        if not halted and os.path.exists(os.path.join(cd, "segment_end.restart")):
            st["start"], st["restart"] = os.path.join(cd, "segment_end.restart"), 1
        rec["t_cycle_s"] = time.time() - t0
        st["student"] = new_student
        st["history"].append(rec)
        st["cycle"] = c + 1
        save_state(wd, st)
        print(f"cycle {c}: rc={rc} halted={halted} checked={len(rows)} worst={worst} density={rec['density_mean']} "
              f"retrained={rec.get('retrained', False)} added={rec.get('added_frames', 0)} ({rec['t_cycle_s']:.0f}s)", flush=True)
    print(f"done: {st['cycle']} cycles, student = {st['student']}", flush=True)


if __name__ == "__main__":
    main()

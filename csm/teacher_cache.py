"""Reuse DIAL rollouts across evaluations of different students.

DIAL is about 99.9% of what an evaluation costs, and it is the *same* DIAL
every time: the teacher is a deterministic function of the objective, the
planner settings, the command, the target weight and the seed -- none of which
mention the student being scored.  Comparing several controllers on one grid
therefore recomputes one fixed set of trajectories once per arm.  A baseline
table with ten arms pays for nine of them twice over.

What makes a cache like this dangerous is not the hit, it is the hit that
should have been a miss.  Every objective change on this project so far has
been a *code* edit -- a reward row deleted, a unit corrected, an observation
narrowed -- and a key built from the config alone would have served the old
numbers under the new objective without a word.  So the key covers three
things, and each of them exists because of a specific way of getting this
wrong:

  the configs      hashed whole, via `dataclasses.asdict`, never a hand-picked
                   subset.  A field added to `DialConfig` or the environment
                   config next month changes the key without anyone
                   remembering to come back here.
  the source       the environment class's own text plus the planner functions
                   that turn returns into an update.  This is the one that
                   catches deleting a reward term.
  the backend      cpu and gpu do not agree to the last bit, and the whole
                   point of the grid is that the student and the teacher are
                   compared on identical states.

A miss is always safe -- it recomputes.  There is nothing here that refuses,
because a key that covers everything makes refusal unnecessary: a changed
objective simply lands in a different directory, and the old one stays valid
for the old code if it is ever checked out again.

Horizons compose the way `--also-report` does.  A short evaluation is a prefix
of a long one, so a cached 1500-step episode answers a 150-step request by
truncation, and a request longer than what is stored is a miss that replaces
it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("csm_runs/_teacher_cache")


def _jsonable(value):
    """Config values as text, arrays included.

    `_load_config` builds the environment config with
    `convert_list_to_array=True`, so reward weights and command bounds arrive
    as numpy arrays and `json.dumps` refuses them.  Rounding is deliberate: a
    float that survived a yaml round trip should not produce a different key
    from the one that did not.
    """

    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return repr(value)


def _digest(payload) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _class_source(env) -> list[str]:
    """The environment class's text, and its bases up to the brax boundary.

    The reward rows and the observation are methods on the class, so this is
    where an objective change actually shows up.  Walking the MRO catches a
    change in a shared base; stopping at anything outside `dial_mpc` or `csm`
    keeps brax's own version out of the key, where the installed package
    version already belongs.
    """

    out = []
    for klass in type(env).__mro__:
        module = getattr(klass, "__module__", "")
        if not (module.startswith("dial_mpc") or module.startswith("csm")):
            continue
        try:
            out.append(inspect.getsource(klass))
        except (OSError, TypeError):
            out.append(f"<unavailable:{module}.{klass.__name__}>")
    return out


def _function_source(functions) -> list[str]:
    out = []
    for fn in functions:
        try:
            out.append(inspect.getsource(fn))
        except (OSError, TypeError):
            out.append(f"<unavailable:{getattr(fn, '__name__', fn)}>")
    return out


def _backend() -> dict:
    import jax

    devices = jax.devices()
    return {
        "platform": devices[0].platform if devices else "none",
        "device_kind": devices[0].device_kind if devices else "none",
        "count": len(devices),
    }


def fingerprint(*, dial_config, env_config, env, functions, extra=None) -> tuple[str, dict]:
    """The cache directory name, and the manifest that explains it.

    Returned together on purpose: the digest is unreadable, so nothing should
    be able to create a directory without also writing down what it stands
    for.
    """

    import jax

    payload = {
        "dial_config": _jsonable(dataclasses.asdict(dial_config)),
        "env_config": _jsonable(dataclasses.asdict(env_config)),
        "env_class": f"{type(env).__module__}.{type(env).__name__}",
        "source": _class_source(env) + _function_source(functions),
        "backend": _backend(),
        "extra": _jsonable(extra or {}),
    }
    digest = _digest(payload)
    manifest = dict(payload)
    # The source text is what makes the key trustworthy and what makes the
    # manifest unreadable.  Keep its digest so a mismatch can be localised,
    # and drop the bodies.
    manifest["source"] = {"sha256": _digest(payload["source"]),
                          "units": len(payload["source"])}
    manifest["jax_version"] = jax.__version__
    manifest["fingerprint"] = digest
    return digest, manifest


def episode_key(*, omega, command, seed) -> str:
    """One episode of the grid.

    Keyed on the weight *vector* and the command *values*, not the names they
    were asked for by: `uniform` and `1,1,1` are the same episode and should
    cost one rollout between them.
    """

    return _digest({
        "omega": [round(float(v), 9) for v in np.asarray(omega).ravel()],
        "command": [round(float(v), 9) for v in np.asarray(command).ravel()],
        "seed": int(seed),
    })


class TeacherCache:
    """Per-episode `(reward, done)` traces under one fingerprint.

    Deliberately not a single archive: separate files mean two evaluations of
    different targets can run at once and fill in different parts of the same
    grid, and an interrupted run keeps whatever it finished.
    """

    def __init__(self, root, digest, manifest, *, refresh: bool = False):
        self.dir = Path(root) / digest
        self.refresh = refresh
        self.hits = 0
        self.misses = 0
        self.stored = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "manifest.json"
        if not path.exists():
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            self._warn_on_drift(path, manifest)

    def _warn_on_drift(self, path, manifest) -> None:
        """Report what changed without invalidating anything.

        Nothing here can change the key -- if it could, this would be a miss
        instead.  These are the parts that are recorded rather than keyed
        because keying them would throw the cache away on every library
        upgrade, so the honest thing is to say so and let the reader judge.
        """

        try:
            stored = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if stored.get("jax_version") != manifest.get("jax_version"):
            print(f"[teacher-cache] note: cached under jax "
                  f"{stored.get('jax_version')}, running "
                  f"{manifest.get('jax_version')}")

    def load(self, key: str, n_steps: int):
        """A stored episode at least `n_steps` long, truncated to fit."""

        if self.refresh:
            return None
        path = self.dir / f"{key}.npz"
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path) as data:
                reward, done = data["reward"], data["done"]
        except (OSError, ValueError, KeyError):
            # A half-written file from a killed run.  Treat it as absent
            # rather than crashing an evaluation that can simply redo it.
            self.misses += 1
            return None
        if reward.shape[0] < n_steps:
            self.misses += 1
            return None
        self.hits += 1
        return reward[:n_steps], done[:n_steps]

    def store(self, key: str, reward, done) -> None:
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        path = self.dir / f"{key}.npz"
        if path.exists():
            try:
                with np.load(path) as data:
                    if data["reward"].shape[0] >= reward.shape[0]:
                        return
            except (OSError, ValueError, KeyError):
                pass
        # Write and rename, so a reader never opens a partial file and a
        # second evaluation writing the same episode cannot interleave.  The
        # temporary name has to end in `.npz` as well: `savez_compressed`
        # appends the extension to anything that does not, and then the rename
        # looks for a file numpy never wrote.
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, reward=reward, done=done)
        os.replace(tmp, path)
        self.stored += 1

    def summary(self) -> str:
        total = self.hits + self.misses
        share = 100.0 * self.hits / total if total else 0.0
        return (f"[teacher-cache] {self.hits}/{total} hits ({share:.0f}%), "
                f"{self.stored} written -> {self.dir}")

# Isaac Sim, Isaac Lab, and Newton Local Setup

Last verified: 2026-05-08.

This machine is set up as a sibling checkout workspace under `/home/horde`.  Isaac Lab runs against a locally built Isaac Sim tree, and its Python environment imports Newton from the local coupled-solver checkout.

## Repo Link Index

Use the exact commits for replication.  The branch links are useful for browsing, but the branch heads can move.

- Isaac Sim source:
  - Private clone URL: `git@gitlab-master.nvidia.com:omniverse/isaac/omni_isaac_sim.git`
  - Private repo: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim`
  - Private branch: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim/-/tree/develop`
  - Private commit: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim/-/commit/fa6c914bc1c3f2e81fe0289aa033e8549db637a6`
  - Public mirror repo: `https://github.com/isaac-sim/IsaacSim`
  - Public mirror branch: `https://github.com/isaac-sim/IsaacSim/tree/develop`

- Active Isaac Lab:
  - Clone URL: `git@github.com:maxkra15/IsaacLab.git`
  - Repo: `https://github.com/maxkra15/IsaacLab`
  - Branch: `https://github.com/maxkra15/IsaacLab/tree/feat/newton-implicit-mpm`
  - Commit: `https://github.com/maxkra15/IsaacLab/commit/12d1d350657ddd821ce016f7dd5cc09f450ed010`

- Isaac Lab PR-5443 reference:
  - Clone URL: `git@github.com:maxkra15/IsaacLab.git`
  - Repo: `https://github.com/maxkra15/IsaacLab`
  - Branch: `https://github.com/maxkra15/IsaacLab/tree/pr-5443-deformable-coupling`
  - Commit: `https://github.com/maxkra15/IsaacLab/commit/89ebfcecfe1fea70a49dfebe663a1d88f95b8754`

- Newton upstream reference:
  - Clone URL: `https://github.com/newton-physics/newton.git`
  - Repo: `https://github.com/newton-physics/newton`
  - Branch: `https://github.com/newton-physics/newton/tree/main`
  - Commit: `https://github.com/newton-physics/newton/commit/9b4069ebafcacf85bc77bb39f96dfb9c245ec40e`

- Active Newton coupled solver:
  - Clone URL: `https://github.com/gdaviet/newton.git`
  - Repo: `https://github.com/gdaviet/newton`
  - Branch: `https://github.com/gdaviet/newton/tree/gdaviet-coupled-solver-framework`
  - Commit: `https://github.com/gdaviet/newton/commit/ba2b8a1baa152fad08dab65984ae8ce2b8073e26`

## Current Checkouts

- `/home/horde/omni_isaac_sim`
  - Purpose: Isaac Sim source checkout and build output.
  - Remote: `git@gitlab-master.nvidia.com:omniverse/isaac/omni_isaac_sim.git`
  - Private repo browser: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim`
  - Public repo browser: `https://github.com/isaac-sim/IsaacSim`
  - Branch: `develop`
  - Branch link: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim/-/tree/develop`
  - Commit: `fa6c914bc1c3f2e81fe0289aa033e8549db637a6`
  - Commit link: `https://gitlab-master.nvidia.com/omniverse/isaac/omni_isaac_sim/-/commit/fa6c914bc1c3f2e81fe0289aa033e8549db637a6`
  - Version file: `6.0.0-alpha.199`
  - Build used by Isaac Lab: `_build/linux-x86_64/release`

- `/home/horde/IsaacLab`
  - Purpose: active Isaac Lab checkout for Newton implicit MPM / proxy-coupling work.
  - Remote: `git@github.com:maxkra15/IsaacLab.git`
  - Repo browser: `https://github.com/maxkra15/IsaacLab`
  - Branch: `feat/newton-implicit-mpm`
  - Commit: `12d1d350657ddd821ce016f7dd5cc09f450ed010`
  - Branch link: `https://github.com/maxkra15/IsaacLab/tree/feat/newton-implicit-mpm`
  - Commit link: `https://github.com/maxkra15/IsaacLab/commit/12d1d350657ddd821ce016f7dd5cc09f450ed010`
  - Local Isaac Sim link: `_isaac_sim -> /home/horde/omni_isaac_sim/_build/linux-x86_64/release`

- `/home/horde/IsaacLab-pr5443`
  - Purpose: separate reference checkout for the deformable-coupling PR branch.
  - Remote: `git@github.com:maxkra15/IsaacLab.git`
  - Repo browser: `https://github.com/maxkra15/IsaacLab`
  - Branch: `pr-5443-deformable-coupling`
  - Commit: `89ebfcecfe1fea70a49dfebe663a1d88f95b8754`
  - Branch link: `https://github.com/maxkra15/IsaacLab/tree/pr-5443-deformable-coupling`
  - Commit link: `https://github.com/maxkra15/IsaacLab/commit/89ebfcecfe1fea70a49dfebe663a1d88f95b8754`
  - Note: this checkout does not currently have the `_isaac_sim` symlink.

- `/home/horde/newton`
  - Purpose: clean upstream Newton reference checkout.
  - Remote: `https://github.com/newton-physics/newton.git`
  - Repo browser: `https://github.com/newton-physics/newton`
  - Branch: `main`
  - Commit: `9b4069ebafcacf85bc77bb39f96dfb9c245ec40e`
  - Branch link: `https://github.com/newton-physics/newton/tree/main`
  - Commit link: `https://github.com/newton-physics/newton/commit/9b4069ebafcacf85bc77bb39f96dfb9c245ec40e`

- `/home/horde/newton-coupled`
  - Purpose: active Newton checkout used by the Isaac Lab environment.
  - Remote: `https://github.com/newton-physics/newton.git`
  - Dependency source repo: `https://github.com/gdaviet/newton.git`
  - Dependency repo browser: `https://github.com/gdaviet/newton`
  - Local branch: `gdaviet-coupled-solver-framework`
  - Commit: `ba2b8a1baa152fad08dab65984ae8ce2b8073e26`
  - Source pinned by active Isaac Lab: `https://github.com/gdaviet/newton.git@ba2b8a1baa152fad08dab65984ae8ce2b8073e26`
  - Branch link: `https://github.com/gdaviet/newton/tree/gdaviet-coupled-solver-framework`
  - Commit link: `https://github.com/gdaviet/newton/commit/ba2b8a1baa152fad08dab65984ae8ce2b8073e26`
  - Dirty state: one local edit in `newton/_src/solvers/coupled/solver_proxy_coupled.py`; see "Local Newton patch" below.

## How the Pieces Are Wired

The active Isaac Lab checkout uses its local `_isaac_sim` symlink to run the Python and Kit binaries from the Isaac Sim build:

```bash
/home/horde/IsaacLab/_isaac_sim -> /home/horde/omni_isaac_sim/_build/linux-x86_64/release
```

The active Python executable is:

```bash
/home/horde/IsaacLab/_isaac_sim/kit/python/bin/python3
```

The installed `newton` package is editable and points at the coupled checkout:

```bash
newton -> /home/horde/newton-coupled
```

The active `IsaacLab/source/isaaclab_newton/setup.py` pins these Newton-side dependencies for the `all` extra:

```text
mujoco==3.8.0
mujoco-warp==3.8.0.1
PyOpenGL-accelerate==3.1.10
newton @ git+https://github.com/gdaviet/newton.git@ba2b8a1baa152fad08dab65984ae8ce2b8073e26
```

Relevant source links:

- Active Isaac Lab `isaaclab_newton/setup.py`: `https://github.com/maxkra15/IsaacLab/blob/feat/newton-implicit-mpm/source/isaaclab_newton/setup.py`
- Active Isaac Lab commit-pinned `isaaclab_newton/setup.py`: `https://github.com/maxkra15/IsaacLab/blob/12d1d350657ddd821ce016f7dd5cc09f450ed010/source/isaaclab_newton/setup.py`
- Active Isaac Lab Newton demos: `https://github.com/maxkra15/IsaacLab/tree/feat/newton-implicit-mpm/scripts/demos`
- Active Isaac Lab UR10 particle-scoop task: `https://github.com/maxkra15/IsaacLab/tree/feat/newton-implicit-mpm/source/isaaclab_tasks/isaaclab_tasks/direct/ur10_particle_scoop`
- Coupled Newton solver source: `https://github.com/gdaviet/newton/blob/ba2b8a1baa152fad08dab65984ae8ce2b8073e26/newton/_src/solvers/coupled/solver_proxy_coupled.py`

The PR-5443 checkout is intentionally different.  Its `isaaclab_newton` package pins upstream Newton at `a27277ed49d6f307b8a1e4c394be7e1d14965a62` with `mujoco==3.6.0` and `mujoco-warp==3.6.0`.

## Replication Steps

Use the same sibling layout unless you also update the symlink paths.

### 1. Clone And Build Isaac Sim

Preferred source for this machine, assuming NVIDIA GitLab access:

```bash
cd /home/horde
git clone -b develop git@gitlab-master.nvidia.com:omniverse/isaac/omni_isaac_sim.git omni_isaac_sim
cd omni_isaac_sim
git checkout fa6c914bc1c3f2e81fe0289aa033e8549db637a6
git lfs install
git lfs pull
./build.sh
```

If the engineer does not have NVIDIA GitLab access, try the public Isaac Sim repository instead:

```bash
cd /home/horde
git clone -b develop https://github.com/isaac-sim/IsaacSim.git omni_isaac_sim
cd omni_isaac_sim
git checkout fa6c914bc1c3f2e81fe0289aa033e8549db637a6 || true
git lfs install
git lfs pull
./build.sh
```

If the public mirror does not have the exact commit, use the nearest `develop` commit that matches Isaac Sim `6.0.0-alpha.199` and retest the smoke commands below.

### 2. Clone Active Isaac Lab

Clone the active Isaac Lab branch and point it at the built Isaac Sim:

```bash
cd /home/horde
git clone git@github.com:maxkra15/IsaacLab.git IsaacLab
cd IsaacLab
git checkout feat/newton-implicit-mpm
git checkout 12d1d350657ddd821ce016f7dd5cc09f450ed010
ln -sfn /home/horde/omni_isaac_sim/_build/linux-x86_64/release _isaac_sim
```

For read-only HTTPS access, use:

```bash
git clone https://github.com/maxkra15/IsaacLab.git IsaacLab
```

Optional: clone the PR-5443 reference branch in a separate directory:

```bash
cd /home/horde
git clone git@github.com:maxkra15/IsaacLab.git IsaacLab-pr5443
cd IsaacLab-pr5443
git checkout pr-5443-deformable-coupling
git checkout 89ebfcecfe1fea70a49dfebe663a1d88f95b8754
ln -sfn /home/horde/omni_isaac_sim/_build/linux-x86_64/release _isaac_sim
```

### 3. Clone Newton Checkouts

Clone the clean upstream reference checkout:

```bash
cd /home/horde
git clone https://github.com/newton-physics/newton.git newton
git -C newton checkout 9b4069ebafcacf85bc77bb39f96dfb9c245ec40e
```

Clone the active coupled-solver checkout from the fork used by the dependency pin:

```bash
cd /home/horde
git clone https://github.com/gdaviet/newton.git newton-coupled
cd newton-coupled
git checkout gdaviet-coupled-solver-framework
git checkout ba2b8a1baa152fad08dab65984ae8ce2b8073e26
```

If the branch name is unavailable but the commit exists, checking out the commit is enough for replication.

### 4. Install Python Packages

Install Isaac Lab into the Isaac Sim Python environment, then override Newton with the local coupled checkout:

```bash
cd /home/horde/IsaacLab
./isaaclab.sh -i all
./_isaac_sim/python.sh -m pip install -e /home/horde/newton-coupled
./_isaac_sim/python.sh -m pip install toml
```

The explicit `toml` install is included because this checkout imports `toml` from `isaaclab_newton/__init__.py`, and the current environment had no standalone `toml` package visible during a direct import sanity check.

## Local Newton Patch

The current `/home/horde/newton-coupled` checkout has one local, uncommitted helper method.  Reapply it if the target commit does not already have it:

```diff
diff --git a/newton/_src/solvers/coupled/solver_proxy_coupled.py b/newton/_src/solvers/coupled/solver_proxy_coupled.py
@@
     def get_proxy_contacts(self, source: str, destination: str) -> Contacts | None:
         config = self._proxy_collision_configs.get((source, destination))
         return None if config is None else config.contacts
 
+    def get_proxy_body_wrenches(self, source: str, destination: str) -> wp.array | None:
+        """Return harvested body feedback wrenches for one proxy direction.
+
+        Values are indexed by parent-model body id and use Newton's spatial-vector
+        layout ``(force_x, force_y, force_z, torque_x, torque_y, torque_z)``.
+        """
+        for mapping in self._proxy_mappings:
+            if mapping.src_name == source and mapping.dst_name == destination:
+                return mapping.coupling_forces
+        return None
+
     def _after_entry_states_created(self) -> None:
         model = self.model
         device = model.device
```

## Sanity Checks

Confirm the local package wiring:

```bash
cd /home/horde/IsaacLab
./isaaclab.sh -p -c "import newton; print(newton.__file__)"
./_isaac_sim/python.sh -m pip show newton isaaclab_newton mujoco mujoco-warp warp-lang
```

Expected `newton.__file__` starts with:

```text
/home/horde/newton-coupled/newton/
```

Run smoke tests and demos:

```bash
cd /home/horde/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit
./isaaclab.sh -p scripts/demos/newton_box_mpm_twoway.py --viz newton --max-steps 100 --disable-cuda-graph
./isaaclab.sh -p scripts/demos/newton_anymal_mpm_sand.py --viz newton --max-steps 100 --disable-cuda-graph
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-UR10-Particle-Scoop-Direct-v0 --num_envs 1 --max_iterations 1 --viz newton
```

## Notes and Caveats

- Isaac Sim `develop` and Isaac Lab `develop` are moving targets.  Prefer the exact commits above for replication instead of only checking out branch tips.
- The active Isaac Lab branch has local Newton MPM demos in `scripts/demos/newton_box_mpm_twoway.py` and `scripts/demos/newton_anymal_mpm_sand.py`.
- The active branch also has the direct UR10 particle-scoop task under `source/isaaclab_tasks/isaaclab_tasks/direct/ur10_particle_scoop`.
- If using `/home/horde/IsaacLab-pr5443`, create its own `_isaac_sim` symlink and install it separately; do not assume the active Isaac Lab editable installs point at that checkout.
- If the machine path is not `/home/horde`, replace the absolute paths in the symlink and editable-install commands.

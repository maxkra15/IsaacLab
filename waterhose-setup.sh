#!/usr/bin/env bash

# Standalone setup helper for the IsaacLab waterhose demo.
#
# Intended use:
#   mkdir -p /path/to/safe/folder
#   cp waterhose-setup.sh /path/to/safe/folder/
#   cp waterhose_demo_assets.tar.gz /path/to/safe/folder/
#   cd /path/to/safe/folder
#   ./waterhose-setup.sh setup --accept-eula
#
# The setup command creates /path/to/safe/folder/waterhose-demo/ and keeps the
# IsaacLab checkout, Isaac Sim source checkout, build, venv, symlink, and assets
# inside that workspace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

DEFAULT_WORKSPACE="${SCRIPT_DIR}/waterhose-demo"
DEFAULT_REPO_DIR_NAME="IsaacLab-waterhose"
DEFAULT_REPO_URL="${WATERHOSE_REPO_URL:-https://github.com/maxkra15/IsaacLab.git}"
DEFAULT_REPO_REF="${WATERHOSE_REPO_REF:-max/waterhose-coupled-experimental}"
DEFAULT_ISAACSIM_DIR_NAME="IsaacSim"
DEFAULT_ISAACSIM_URL="${ISAACSIM_REPO_URL:-https://github.com/isaac-sim/IsaacSim.git}"
DEFAULT_ISAACSIM_REF="${ISAACSIM_REPO_REF:-develop}"
DEFAULT_VENV=".venv"
DEFAULT_ASSETS_TAR="${SCRIPT_DIR}/waterhose_demo_assets.tar.gz"
DEFAULT_TASK="Isaac-Waterhose-Coupled-v0"

log() {
    printf '[waterhose] %s\n' "$*"
}

warn() {
    printf '[waterhose:warning] %s\n' "$*" >&2
}

die() {
    printf '[waterhose:error] %s\n' "$*" >&2
    exit 1
}

run() {
    log "+ $*"
    "$@"
}

# Clone a repo and check out a branch, tag, or commit SHA. `git clone --branch` only accepts a
# branch or tag, so fall back to a plain clone followed by an explicit fetch + checkout for SHAs.
clone_ref() {
    local url="$1"
    local dir="$2"
    local ref="$3"
    if run git clone --branch "$ref" "$url" "$dir" 2>/dev/null; then
        return
    fi
    log "Ref '${ref}' is not a branch/tag; cloning and checking it out explicitly."
    run git clone "$url" "$dir"
    run git -C "$dir" fetch origin "$ref"
    git -C "$dir" checkout "$ref" 2>/dev/null || run git -C "$dir" checkout FETCH_HEAD
}

source_without_nounset() {
    local script_path="$1"
    local had_nounset=0

    case "$-" in
        *u*)
            had_nounset=1
            set +u
            ;;
    esac

    # shellcheck disable=SC1090
    source "$script_path"

    if [[ "$had_nounset" == "1" ]]; then
        set -u
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./waterhose-setup.sh setup [options]
  ./waterhose-setup.sh init [options]
  ./waterhose-setup.sh help

Fresh-machine setup:
  ./waterhose-setup.sh setup --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz
  # init is an alias for setup:
  ./waterhose-setup.sh init --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz

What setup creates by default:
  ./waterhose-demo/
    IsaacLab-waterhose/       IsaacLab waterhose checkout and .venv
    IsaacSim/                 Isaac Sim source checkout and build

Clean setup behavior:
  setup aborts if the workspace already contains files, because mixing an old
  Isaac Sim build, venv, or IsaacLab checkout can create hard-to-debug issues.
  Use --resume-existing only when intentionally continuing a known partial setup.

The default IsaacLab repo URL points at a development fork. For a supported handoff, set
--repo-url / --repo-ref (or WATERHOSE_REPO_URL / WATERHOSE_REPO_REF) to the branch, tag, or
commit your NVIDIA contact provided. --repo-ref and --isaacsim-ref accept a branch, tag, or
commit SHA.

Setup options:
  --workspace DIR            Workspace to create. Default: ./waterhose-demo
  --repo-url URL             IsaacLab waterhose repo URL (default is a development fork).
  --repo-ref REF             IsaacLab waterhose branch/tag/commit. Default: max/waterhose-coupled-experimental
  --repo-dir-name NAME       Checkout dir inside workspace. Default: IsaacLab-waterhose
  --isaacsim-url URL         Isaac Sim repo URL.
  --isaacsim-ref REF         Isaac Sim branch/tag/commit. Default: develop
  --isaacsim-dir-name NAME   Isaac Sim dir inside workspace. Default: IsaacSim
  --venv DIR                 venv directory inside IsaacLab checkout. Default: .venv
  --assets-tar FILE          waterhose_demo_assets.tar.gz path.
  --accept-eula              Non-interactively accept Isaac Sim additional terms.
  --jobs N                   Pass -j N to Isaac Sim build.sh.
  --build-arg ARG            Add one argument to Isaac Sim build.sh.
  --all-configs              Build Isaac Sim default configs instead of release only.
  --resume-existing          Continue from an existing workspace instead of aborting.
  --rebuild-isaacsim         Rebuild Isaac Sim when resuming an existing workspace.
  --skip-host-deps           Do not install apt packages.
  --skip-gcc-alternatives    Do not set gcc/g++ alternatives to version 11.
  --skip-lfs                 Do not run git lfs install/pull.
  --skip-smoke               Do not run the post-install headless smoke check.

Setup installs the full Isaac Lab workspace with `isaaclab.sh -i all`. Runtime
commands are intentionally documented in docs/waterhose_robot_demo.md instead
of being hidden behind this setup script.
EOF
}

abs_path() {
    local path="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "$path"
    elif [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$PWD" "$path"
    fi
}

repo_dir_for_workspace() {
    local workspace="$1"
    local repo_dir_name="${2:-$DEFAULT_REPO_DIR_NAME}"
    printf '%s/%s\n' "$workspace" "$repo_dir_name"
}

isaacsim_dir_for_workspace() {
    local workspace="$1"
    local isaacsim_dir_name="${2:-$DEFAULT_ISAACSIM_DIR_NAME}"
    printf '%s/%s\n' "$workspace" "$isaacsim_dir_name"
}

venv_path() {
    local repo_root="$1"
    local venv_name="$2"
    if [[ "$venv_name" = /* ]]; then
        printf '%s\n' "$venv_name"
    else
        printf '%s/%s\n' "$repo_root" "$venv_name"
    fi
}

sudo_cmd() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install host packages. Use --skip-host-deps only if you installed them manually."
        sudo "$@"
    fi
}

require_linux() {
    [[ "$(uname -s)" == "Linux" ]] || die "This helper currently supports Linux source builds only."
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

install_host_deps() {
    require_linux
    if ! command -v apt-get >/dev/null 2>&1; then
        die "apt-get was not found. Install host dependencies manually and re-run with --skip-host-deps."
    fi

    local packages=(
        build-essential
        ca-certificates
        cmake
        curl
        g++-11
        gcc-11
        git
        git-lfs
        python3
        python3-dev
        python3-venv
    )

    case "$(uname -m)" in
        aarch64|arm64)
            packages+=(libx11-dev xorg-dev swig)
            ;;
    esac

    run sudo_cmd apt-get update
    run sudo_cmd apt-get install -y --no-install-recommends "${packages[@]}"
}

ensure_uv() {
    export PATH="${HOME}/.local/bin:${PATH}"
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    require_command curl
    log "Installing uv into ${HOME}/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    require_command uv
}

configure_gcc_11() {
    require_linux
    if ! command -v gcc-11 >/dev/null 2>&1 || ! command -v g++-11 >/dev/null 2>&1; then
        die "gcc-11/g++-11 are required for the Isaac Sim source build."
    fi

    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

    if command -v update-alternatives >/dev/null 2>&1; then
        run sudo_cmd update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
        run sudo_cmd update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
        run sudo_cmd update-alternatives --set gcc /usr/bin/gcc-11
        run sudo_cmd update-alternatives --set g++ /usr/bin/g++-11
    else
        warn "update-alternatives was not found. Exported CC/CXX to gcc-11/g++-11 only."
    fi
}

workspace_has_content() {
    local workspace="$1"
    [[ -d "$workspace" ]] || return 1
    find "$workspace" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

prepare_clean_workspace() {
    local workspace="$1"
    local resume_existing="$2"

    if workspace_has_content "$workspace"; then
        if [[ "$resume_existing" == "1" ]]; then
            log "Reusing existing workspace: ${workspace}"
            return
        fi
        die "Workspace already exists and is not empty: ${workspace}. Remove it or re-run with --resume-existing."
    fi

    mkdir -p "$workspace"
}

ensure_clean_git_checkout() {
    local repo_dir="$1"
    [[ -d "${repo_dir}/.git" ]] || return
    if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
        die "Git checkout has local changes: ${repo_dir}. Commit/stash them before using --resume-existing."
    fi
}

clone_or_resume_repo() {
    local repo_dir="$1"
    local repo_url="$2"
    local repo_ref="$3"
    local resume_existing="$4"

    if [[ -e "$repo_dir" && ! -d "${repo_dir}/.git" ]]; then
        die "Repo path exists but is not a git checkout: ${repo_dir}"
    fi

    if [[ -d "${repo_dir}/.git" ]]; then
        [[ "$resume_existing" == "1" ]] || die "Repo checkout already exists: ${repo_dir}"
        reset_newton_git_ref_patches "$repo_dir"
        ensure_clean_git_checkout "$repo_dir"
        run git -C "$repo_dir" fetch origin --tags --prune
        if git -C "$repo_dir" show-ref --verify --quiet "refs/remotes/origin/${repo_ref}"; then
            run git -C "$repo_dir" checkout -B "$repo_ref" "origin/${repo_ref}"
        else
            run git -C "$repo_dir" checkout "$repo_ref"
        fi
        return
    fi

    clone_ref "$repo_url" "$repo_dir" "$repo_ref"
}

clone_or_resume_isaacsim() {
    local isaacsim_dir="$1"
    local isaacsim_url="$2"
    local isaacsim_ref="$3"
    local resume_existing="$4"
    local skip_lfs="$5"

    if [[ -e "$isaacsim_dir" && ! -d "${isaacsim_dir}/.git" ]]; then
        die "Isaac Sim path exists but is not a git checkout: ${isaacsim_dir}"
    fi

    if [[ -d "${isaacsim_dir}/.git" ]]; then
        [[ "$resume_existing" == "1" ]] || die "Isaac Sim checkout already exists: ${isaacsim_dir}"
        ensure_clean_git_checkout "$isaacsim_dir"
        run git -C "$isaacsim_dir" fetch origin --tags --prune
        if git -C "$isaacsim_dir" show-ref --verify --quiet "refs/remotes/origin/${isaacsim_ref}"; then
            run git -C "$isaacsim_dir" checkout -B "$isaacsim_ref" "origin/${isaacsim_ref}"
        else
            run git -C "$isaacsim_dir" checkout "$isaacsim_ref"
        fi
    else
        # Skip LFS smudge during the clone; LFS objects are pulled below unless --skip-lfs.
        (export GIT_LFS_SKIP_SMUDGE=1 && clone_ref "$isaacsim_url" "$isaacsim_dir" "$isaacsim_ref")
    fi

    if [[ "$skip_lfs" != "1" ]]; then
        run git -C "$isaacsim_dir" lfs install
        run git -C "$isaacsim_dir" lfs pull
    fi
}

accept_isaacsim_eula_if_requested() {
    local isaacsim_dir="$1"
    local accept_eula="$2"

    if [[ -f "${isaacsim_dir}/.eula_accepted" ]]; then
        return
    fi
    if [[ "$accept_eula" == "1" || "${ACCEPT_NVIDIA_EULA:-0}" == "1" ]]; then
        log "Recording Isaac Sim EULA acceptance at ${isaacsim_dir}/.eula_accepted"
        : > "${isaacsim_dir}/.eula_accepted"
        return
    fi
    if [[ -t 0 ]]; then
        warn "Isaac Sim will prompt for EULA acceptance during build. Use --accept-eula for non-interactive setup."
        return
    fi
    die "Isaac Sim EULA acceptance is required in non-interactive setup. Re-run with --accept-eula after reviewing the terms."
}

isaacsim_build_dir() {
    local isaacsim_dir="$1"
    case "$(uname -m)" in
        x86_64|amd64)
            printf '%s/_build/linux-x86_64/release\n' "$isaacsim_dir"
            ;;
        aarch64|arm64)
            printf '%s/_build/linux-aarch64/release\n' "$isaacsim_dir"
            ;;
        *)
            die "Unsupported architecture for Isaac Sim source build: $(uname -m)"
            ;;
    esac
}

isaacsim_build_ready() {
    local isaacsim_dir="$1"
    local build_dir
    build_dir="$(isaacsim_build_dir "$isaacsim_dir")"
    [[ -d "$build_dir" && -f "${build_dir}/python.sh" ]] || return 1
    [[ -f "${build_dir}/setup_conda_env.sh" || -f "${build_dir}/setup_python_env.sh" ]]
}

build_isaacsim() {
    local isaacsim_dir="$1"
    shift
    local build_args=("$@")

    [[ -x "${isaacsim_dir}/build.sh" ]] || die "Isaac Sim build.sh not found at ${isaacsim_dir}/build.sh"
    export CC="${CC:-/usr/bin/gcc-11}"
    export CXX="${CXX:-/usr/bin/g++-11}"
    log "Building Isaac Sim. This can take a while."
    (cd "$isaacsim_dir" && run ./build.sh "${build_args[@]}")
}

isaacsim_has_wheeled_robots() {
    local build_dir="$1"
    [[ -f "${build_dir}/exts/isaacsim.robot.wheeled_robots/config/extension.toml" ]] && return 0
    [[ -f "${build_dir}/exts/isaacsim.robot.experimental.wheeled_robots/config/extension.toml" ]] && return 0
    [[ -f "${build_dir}/extsDeprecated/isaacsim.robot.wheeled_robots/config/extension.toml" ]] && return 0
    return 1
}

ensure_isaacsim_setup_conda_env() {
    local build_dir="$1"
    local setup_conda_env="${build_dir}/setup_conda_env.sh"
    local setup_python_env="${build_dir}/setup_python_env.sh"

    if [[ -f "$setup_conda_env" ]]; then
        return
    fi
    [[ -f "$setup_python_env" ]] || die "Isaac Sim setup_conda_env.sh and setup_python_env.sh not found in ${build_dir}"

    log "Creating Isaac Sim setup_conda_env.sh compatibility shim in ${build_dir}"
    cat > "$setup_conda_env" <<'EOF'
#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MY_DIR="$(realpath -s "$SCRIPT_DIR")"

export CARB_APP_PATH="$SCRIPT_DIR/kit"
export EXP_PATH="$MY_DIR/apps"
export ISAAC_PATH="$MY_DIR"

_WATERHOSE_OLDPWD="$PWD"
cd "$SCRIPT_DIR"
. ./setup_python_env.sh
cd "$_WATERHOSE_OLDPWD"
unset _WATERHOSE_OLDPWD

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "$SCRIPT_DIR/kit/python/lib/python3.12" | tr '\n' ':' | sed 's/:$//')"
fi
EOF
    chmod +x "$setup_conda_env"
}

link_isaacsim_build() {
    local repo_root="$1"
    local isaacsim_dir="$2"
    local build_dir
    build_dir="$(isaacsim_build_dir "$isaacsim_dir")"

    [[ -d "$build_dir" ]] || die "Expected Isaac Sim build directory does not exist: ${build_dir}"
    [[ -f "${build_dir}/python.sh" ]] || die "Isaac Sim python.sh not found in ${build_dir}"
    ensure_isaacsim_setup_conda_env "$build_dir"
    [[ -f "${build_dir}/setup_conda_env.sh" ]] || die "Isaac Sim setup_conda_env.sh not found in ${build_dir}"
    isaacsim_has_wheeled_robots "$build_dir" || die "Isaac Sim build is missing wheeled robot extensions (isaacsim.robot.wheeled_robots or isaacsim.robot.experimental.wheeled_robots). Rebuild Isaac Sim or use a compatible --isaacsim-ref."

    ln -sfn "$build_dir" "${repo_root}/_isaac_sim"
    log "Linked ${repo_root}/_isaac_sim -> ${build_dir}"
}

newton_git_ref_relpaths() {
    printf '%s\n' \
        "source/isaaclab_newton/pyproject.toml" \
        "source/isaaclab_physx/pyproject.toml" \
        "source/isaaclab_visualizers/pyproject.toml" \
        "tools/wheel_builder/res/python_packages.toml"
}

reset_newton_git_ref_patches() {
    local repo_root="$1"
    local rel

    [[ -d "${repo_root}/.git" ]] || return
    while IFS= read -r rel; do
        [[ -e "${repo_root}/${rel}" ]] || continue
        git -C "$repo_root" checkout -- "$rel" 2>/dev/null || true
    done < <(newton_git_ref_relpaths)
}

patch_isaaclab_kit_ext_folders() {
    local repo_root="$1"
    local kit_file="${repo_root}/apps/isaaclab.python.kit"
    local marker='${exe-path}/../extsDeprecated'

    [[ -f "$kit_file" ]] || return
    if grep -qF "$marker" "$kit_file"; then
        return
    fi

    sed -i '/\${exe-path}\/\.\.\/exts",  # isaac extensions/a\
    "${exe-path}/../extsDeprecated",  # deprecated isaac extensions still referenced by Isaac Lab' "$kit_file"
    log "Patched ${kit_file} to search extsDeprecated (needed for isaacsim.sensors.rtx on Isaac Sim develop)"
}

patch_newton_git_refs() {
    local repo_root="$1"
    local old_ref="526b36396777c18b82af8f30c4693b7c8bb4d89d"
    local new_ref="refs/pull/2848/head"
    local rel file patched=0

    while IFS= read -r rel; do
        file="${repo_root}/${rel}"
        [[ -f "$file" ]] || continue
        if grep -q "$old_ref" "$file"; then
            sed -i "s|${old_ref}|${new_ref}|g" "$file"
            patched=1
        fi
    done < <(newton_git_ref_relpaths)

    if [[ "$patched" == "1" ]]; then
        log "Patched Newton git refs to ${new_ref} (uv cannot fetch PR-only commits by SHA)"
    fi
}

setup_uv_env_and_install_isaaclab() {
    local repo_root="$1"
    local venv_name="$2"
    local resume_existing="$3"
    local venv_dir
    venv_dir="$(venv_path "$repo_root" "$venv_name")"

    ensure_uv
    [[ -e "${repo_root}/_isaac_sim" ]] || die "_isaac_sim is missing in ${repo_root}"
    if [[ -e "$venv_dir" ]]; then
        if [[ "$resume_existing" != "1" ]]; then
            die "Virtual environment already exists: ${venv_dir}. Remove the workspace or use --resume-existing only for a known partial setup."
        fi
        log "Reusing existing virtual environment: ${venv_dir}"
    else
        run env -u VIRTUAL_ENV -u CONDA_PREFIX "${repo_root}/isaaclab.sh" --uv "$venv_name"
    fi

    source_without_nounset "${venv_dir}/bin/activate"
    run "${repo_root}/isaaclab.sh" -i all
}

extract_assets() {
    local repo_root="$1"
    local assets_tar="$2"
    local resume_existing="$3"
    local target_dir="${repo_root}/source/isaaclab_assets/data"
    local expected_dir="${target_dir}/WaterhoseDemo"

    if [[ -d "$expected_dir" ]]; then
        if [[ "$resume_existing" == "1" ]]; then
            log "Waterhose assets already present at ${expected_dir}"
            return
        fi
        die "Waterhose assets already exist at ${expected_dir}. Remove the workspace or use a clean setup."
    fi
    [[ -f "$assets_tar" ]] || die "Asset archive not found: ${assets_tar}. Copy waterhose_demo_assets.tar.gz next to this script or pass --assets-tar FILE."

    mkdir -p "$target_dir"
    log "Unpacking waterhose assets into ${target_dir}"
    tar -xzf "$assets_tar" -C "$target_dir"
    [[ -d "$expected_dir" ]] || die "Asset archive did not create ${expected_dir}"
}

run_setup_smoke_check() {
    local repo_root="$1"
    local venv_name="$2"
    local venv_dir
    venv_dir="$(venv_path "$repo_root" "$venv_name")"
    local python_exe="${venv_dir}/bin/python"
    [[ -x "$python_exe" ]] || die "Python executable not found: ${python_exe}"

    log "Running post-install smoke check. Use --skip-smoke to skip this step."
    # Run through isaaclab.sh -- the same entry point documented for customers -- so the smoke
    # check exercises the venv selection and Isaac Sim environment setup the wrapper performs.
    (
        cd "$repo_root"
        export VIRTUAL_ENV="$venv_dir"
        export PATH="${venv_dir}/bin:${PATH}"
        run "${repo_root}/isaaclab.sh" -p \
            scripts/environments/waterhose/run_robot_demo.py \
            --task "$DEFAULT_TASK" \
            --num_envs 1 \
            --max_steps 20 \
            --visualizer none \
            --profile
    )
}

cmd_setup() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_url="$DEFAULT_REPO_URL"
    local repo_ref="$DEFAULT_REPO_REF"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local isaacsim_url="$DEFAULT_ISAACSIM_URL"
    local isaacsim_ref="$DEFAULT_ISAACSIM_REF"
    local isaacsim_dir_name="$DEFAULT_ISAACSIM_DIR_NAME"
    local venv_name="$DEFAULT_VENV"
    local assets_tar="$DEFAULT_ASSETS_TAR"
    local accept_eula=0
    local resume_existing=0
    local skip_host_deps=0
    local skip_gcc_alternatives=0
    local skip_lfs=0
    local skip_smoke=0
    local rebuild_isaacsim=0
    local release_only=1
    local build_args=()

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-url)
                repo_url="$2"; shift 2 ;;
            --repo-ref)
                repo_ref="$2"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --isaacsim-url)
                isaacsim_url="$2"; shift 2 ;;
            --isaacsim-ref)
                isaacsim_ref="$2"; shift 2 ;;
            --isaacsim-dir-name)
                isaacsim_dir_name="$2"; shift 2 ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --assets-tar)
                assets_tar="$(abs_path "$2")"; shift 2 ;;
            --accept-eula)
                accept_eula=1; shift ;;
            --jobs|-j)
                build_args+=("-j" "$2"); shift 2 ;;
            --build-arg)
                build_args+=("$2"); shift 2 ;;
            --all-configs)
                release_only=0; shift ;;
            --resume-existing)
                resume_existing=1; shift ;;
            --rebuild-isaacsim)
                rebuild_isaacsim=1; shift ;;
            --skip-host-deps)
                skip_host_deps=1; shift ;;
            --skip-gcc-alternatives)
                skip_gcc_alternatives=1; shift ;;
            --skip-lfs)
                skip_lfs=1; shift ;;
            --skip-smoke)
                skip_smoke=1; shift ;;
            --help|-h)
                usage; exit 0 ;;
            --)
                shift
                build_args+=("$@")
                break ;;
            *)
                die "Unknown setup option: $1" ;;
        esac
    done

    require_linux
    prepare_clean_workspace "$workspace" "$resume_existing"

    if [[ "$skip_host_deps" != "1" ]]; then
        install_host_deps
    else
        require_command git
        require_command tar
    fi

    ensure_uv
    if [[ "$skip_gcc_alternatives" != "1" ]]; then
        configure_gcc_11
    fi

    local repo_root
    local isaacsim_dir
    repo_root="$(repo_dir_for_workspace "$workspace" "$repo_dir_name")"
    isaacsim_dir="$(isaacsim_dir_for_workspace "$workspace" "$isaacsim_dir_name")"

    if [[ "$repo_url" == "$DEFAULT_REPO_URL" ]]; then
        warn "Using the default IsaacLab repo URL (${repo_url}), which is a development fork. For a"
        warn "supported handoff, pass --repo-url / --repo-ref (or WATERHOSE_REPO_URL / WATERHOSE_REPO_REF)."
    fi

    clone_or_resume_repo "$repo_root" "$repo_url" "$repo_ref" "$resume_existing"
    patch_newton_git_refs "$repo_root"
    patch_isaaclab_kit_ext_folders "$repo_root"
    clone_or_resume_isaacsim "$isaacsim_dir" "$isaacsim_url" "$isaacsim_ref" "$resume_existing" "$skip_lfs"
    accept_isaacsim_eula_if_requested "$isaacsim_dir" "$accept_eula"

    if [[ "$release_only" == "1" ]]; then
        build_args=("-r" "${build_args[@]}")
    fi
    if [[ "$resume_existing" == "1" && "$rebuild_isaacsim" != "1" ]] && isaacsim_build_ready "$isaacsim_dir"; then
        log "Reusing existing Isaac Sim build: $(isaacsim_build_dir "$isaacsim_dir")"
    else
        build_isaacsim "$isaacsim_dir" "${build_args[@]}"
    fi
    link_isaacsim_build "$repo_root" "$isaacsim_dir"

    setup_uv_env_and_install_isaaclab "$repo_root" "$venv_name" "$resume_existing"
    extract_assets "$repo_root" "$assets_tar" "$resume_existing"

    if [[ "$skip_smoke" != "1" ]]; then
        run_setup_smoke_check "$repo_root" "$venv_name"
    fi

    log "Setup complete."
    log "Workspace: ${workspace}"
}

main() {
    local command="${1:-help}"
    [[ $# -gt 0 ]] && shift || true

    case "$command" in
        setup|init)
            cmd_setup "$@" ;;
        help|--help|-h)
            usage ;;
        *)
            die "Unknown command: ${command}. Run ./waterhose-setup.sh help." ;;
    esac
}

if [[ "${WATERHOSE_SETUP_SKIP_MAIN:-0}" != "1" ]]; then
    main "$@"
fi

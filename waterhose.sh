#!/usr/bin/env bash

# Standalone bootstrap and run helper for the IsaacLab waterhose demo.
#
# Intended use:
#   mkdir -p /path/to/safe/folder
#   cp waterhose.sh /path/to/safe/folder/
#   cp waterhose_demo_assets.tar.gz /path/to/safe/folder/
#   cd /path/to/safe/folder
#   ./waterhose.sh setup --accept-eula
#
# The setup command creates /path/to/safe/folder/waterhose-demo/ and keeps the
# IsaacLab checkout, Isaac Sim source checkout, build, venv, symlink, and assets
# inside that workspace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

DEFAULT_WORKSPACE="${SCRIPT_DIR}/waterhose-demo"
DEFAULT_REPO_DIR_NAME="IsaacLab-waterhose"
DEFAULT_REPO_URL="${WATERHOSE_REPO_URL:-https://github.com/maxkra15/IsaacLab.git}"
DEFAULT_REPO_REF="${WATERHOSE_REPO_REF:-waterhose-demo}"
DEFAULT_ISAACSIM_DIR_NAME="IsaacSim"
DEFAULT_ISAACSIM_URL="${ISAACSIM_REPO_URL:-https://github.com/isaac-sim/IsaacSim.git}"
DEFAULT_ISAACSIM_REF="${ISAACSIM_REPO_REF:-develop}"
DEFAULT_VENV=".venv"
DEFAULT_ASSETS_TAR="${SCRIPT_DIR}/waterhose_demo_assets.tar.gz"
DEFAULT_TASK="Isaac-Waterhose-Robot-Demo-v0"
DEFAULT_MIMIC_TASK="Isaac-Waterhose-Robot-Demo-Mimic-v0"

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

usage() {
    cat <<'EOF'
Usage:
  ./waterhose.sh setup [options]
  ./waterhose.sh demo [options] [-- extra demo args]
  ./waterhose.sh teleop [options] [-- extra teleop args]
  ./waterhose.sh smoke [options]
  ./waterhose.sh profile [options]
  ./waterhose.sh mimic-smoke [options]
  ./waterhose.sh python [--workspace DIR] <script.py> [args...]
  ./waterhose.sh shell [--workspace DIR]

Fresh-machine setup:
  ./waterhose.sh setup --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz

What setup creates by default:
  ./waterhose-demo/
    IsaacLab-waterhose/       IsaacLab waterhose checkout and .venv
    IsaacSim/                 Isaac Sim source checkout and build

Clean setup behavior:
  setup aborts if the workspace already contains files, because mixing an old
  Isaac Sim build, venv, or IsaacLab checkout can create hard-to-debug issues.
  Use --resume-existing only when intentionally continuing a known partial setup.

Setup options:
  --workspace DIR            Workspace to create. Default: ./waterhose-demo
  --repo-url URL             IsaacLab waterhose repo URL.
  --repo-ref REF             IsaacLab waterhose branch/tag. Default: waterhose-demo
  --repo-dir-name NAME       Checkout dir inside workspace. Default: IsaacLab-waterhose
  --isaacsim-url URL         Isaac Sim repo URL.
  --isaacsim-ref REF         Isaac Sim branch/tag. Default: develop
  --isaacsim-dir-name NAME   Isaac Sim dir inside workspace. Default: IsaacSim
  --venv DIR                 venv directory inside IsaacLab checkout. Default: .venv
  --assets-tar FILE          waterhose_demo_assets.tar.gz path.
  --accept-eula              Non-interactively accept Isaac Sim additional terms.
  --jobs N                   Pass -j N to Isaac Sim build.sh.
  --build-arg ARG            Add one argument to Isaac Sim build.sh.
  --all-configs              Build Isaac Sim default configs instead of release only.
  --resume-existing          Continue from an existing workspace instead of aborting.
  --skip-host-deps           Do not install apt packages.
  --skip-gcc-alternatives    Do not set gcc/g++ alternatives to version 11.
  --skip-lfs                 Do not run git lfs install/pull.
  --skip-smoke               Do not run the post-install headless smoke check.

Demo options:
  --workspace DIR            Workspace created by setup. Default: ./waterhose-demo
  --task TASK                Task name. Default: Isaac-Waterhose-Robot-Demo-v0
  --vis VALUE                kit, newton, kit,newton, or none. Default: kit
  --num-envs N               Number of environments. Default: 1
  --max-steps N              Demo step limit. Default: 2000
  --headless                 Pass --headless to IsaacLab.
  --profile                  Print timing from the runner.
  --teleop                   Use the demo runner's built-in SpaceMouse teleop mode.
  --venv DIR                 venv directory inside IsaacLab checkout. Default: .venv

Teleop options:
  Uses scripts/environments/teleoperation/teleop_se3_agent.py.

  --workspace DIR            Workspace created by setup. Default: ./waterhose-demo
  --teleop-device NAME       keyboard, spacemouse, or gamepad. Default: spacemouse
  --isaac-teleop             Use env_cfg.isaac_teleop instead of legacy teleop_device.
  --xr                       Use IsaacTeleop/OpenXR mode. Implies --isaac-teleop.
  --cloudxr-env VALUE        cloudxrjs, avp, none, or a .env path. Default: cloudxrjs
  --no-auto-launch-cloudxr   Do not auto-launch CloudXR.
  --task TASK                Task name. Default: Isaac-Waterhose-Robot-Demo-v0
  --vis VALUE                kit or kit,newton. Default: kit
  --num-envs N               Number of environments. Default: 1
  --sensitivity VALUE        Teleop sensitivity. Default: 1.0
  --debug-teleop             Print periodic teleop diagnostics.
  --venv DIR                 venv directory inside IsaacLab checkout. Default: .venv

Examples:
  ./waterhose.sh setup --accept-eula
  ./waterhose.sh demo --vis kit,newton --num-envs 2
  ./waterhose.sh demo --vis none --headless --profile --max-steps 500
  ./waterhose.sh teleop --teleop-device spacemouse --vis kit
  ./waterhose.sh teleop --xr --cloudxr-env avp --vis kit
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
        ensure_clean_git_checkout "$repo_dir"
        run git -C "$repo_dir" fetch origin --tags --prune
        if git -C "$repo_dir" show-ref --verify --quiet "refs/remotes/origin/${repo_ref}"; then
            run git -C "$repo_dir" checkout -B "$repo_ref" "origin/${repo_ref}"
        else
            run git -C "$repo_dir" checkout "$repo_ref"
        fi
        return
    fi

    run git clone --branch "$repo_ref" "$repo_url" "$repo_dir"
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
        run env GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$isaacsim_ref" "$isaacsim_url" "$isaacsim_dir"
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

link_isaacsim_build() {
    local repo_root="$1"
    local isaacsim_dir="$2"
    local build_dir
    build_dir="$(isaacsim_build_dir "$isaacsim_dir")"

    [[ -d "$build_dir" ]] || die "Expected Isaac Sim build directory does not exist: ${build_dir}"
    [[ -f "${build_dir}/python.sh" ]] || die "Isaac Sim python.sh not found in ${build_dir}"
    [[ -f "${build_dir}/setup_conda_env.sh" ]] || die "Isaac Sim setup_conda_env.sh not found in ${build_dir}"

    ln -sfn "$build_dir" "${repo_root}/_isaac_sim"
    log "Linked ${repo_root}/_isaac_sim -> ${build_dir}"
}

setup_uv_env_and_install_isaaclab() {
    local repo_root="$1"
    local venv_name="$2"
    local venv_dir
    venv_dir="$(venv_path "$repo_root" "$venv_name")"

    ensure_uv
    [[ -e "${repo_root}/_isaac_sim" ]] || die "_isaac_sim is missing in ${repo_root}"
    if [[ -e "$venv_dir" ]]; then
        die "Virtual environment already exists: ${venv_dir}. Remove the workspace or use --resume-existing only for a known partial setup."
    fi

    run "${repo_root}/isaaclab.sh" --uv "$venv_name"
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
    run "${repo_root}/isaaclab.sh" -i all
}

extract_assets() {
    local repo_root="$1"
    local assets_tar="$2"
    local target_dir="${repo_root}/source/isaaclab_assets/data"
    local expected_dir="${target_dir}/WaterhoseDemo"

    if [[ -d "$expected_dir" ]]; then
        die "Waterhose assets already exist at ${expected_dir}. Remove the workspace or use a clean setup."
    fi
    [[ -f "$assets_tar" ]] || die "Asset archive not found: ${assets_tar}. Copy waterhose_demo_assets.tar.gz next to this script or pass --assets-tar FILE."

    mkdir -p "$target_dir"
    log "Unpacking waterhose assets into ${target_dir}"
    tar -xzf "$assets_tar" -C "$target_dir"
    [[ -d "$expected_dir" ]] || die "Asset archive did not create ${expected_dir}"
}

resolve_repo_root() {
    local workspace="$1"
    local repo_dir_name="${2:-$DEFAULT_REPO_DIR_NAME}"
    local repo_root
    repo_root="$(repo_dir_for_workspace "$workspace" "$repo_dir_name")"
    [[ -d "$repo_root" ]] || die "IsaacLab checkout not found: ${repo_root}. Run ./waterhose.sh setup first."
    [[ -x "${repo_root}/isaaclab.sh" ]] || die "isaaclab.sh not found or not executable in ${repo_root}"
    printf '%s\n' "$repo_root"
}

activate_runtime_env() {
    local repo_root="$1"
    local venv_name="$2"
    local venv_dir
    venv_dir="$(venv_path "$repo_root" "$venv_name")"
    local activate="${venv_dir}/bin/activate"
    [[ -f "$activate" ]] || die "Virtual environment not found: ${venv_dir}. Run ./waterhose.sh setup first."
    # shellcheck disable=SC1090
    source "$activate"
    if [[ -f "${repo_root}/_isaac_sim/setup_conda_env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${repo_root}/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    fi
    export PYTHONPATH="${repo_root}/source/isaaclab:${PYTHONPATH:-}"
}

run_isaaclab_python() {
    local repo_root="$1"
    local venv_name="$2"
    shift 2
    activate_runtime_env "$repo_root" "$venv_name"
    (cd "$repo_root" && exec "${repo_root}/isaaclab.sh" -p "$@")
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

    clone_or_resume_repo "$repo_root" "$repo_url" "$repo_ref" "$resume_existing"
    clone_or_resume_isaacsim "$isaacsim_dir" "$isaacsim_url" "$isaacsim_ref" "$resume_existing" "$skip_lfs"
    accept_isaacsim_eula_if_requested "$isaacsim_dir" "$accept_eula"

    if [[ "$release_only" == "1" ]]; then
        build_args=("-r" "${build_args[@]}")
    fi
    build_isaacsim "$isaacsim_dir" "${build_args[@]}"
    link_isaacsim_build "$repo_root" "$isaacsim_dir"

    setup_uv_env_and_install_isaaclab "$repo_root" "$venv_name"
    extract_assets "$repo_root" "$assets_tar"

    if [[ "$skip_smoke" != "1" ]]; then
        cmd_smoke --workspace "$workspace" --repo-dir-name "$repo_dir_name" --venv "$venv_name"
    fi

    log "Setup complete."
    log "Workspace: ${workspace}"
}

cmd_demo() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local task="$DEFAULT_TASK"
    local vis="kit"
    local num_envs=1
    local max_steps=2000
    local mode="scripted"
    local venv_name="$DEFAULT_VENV"
    local headless=0
    local profile=0
    local extra=()

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --task)
                task="$2"; shift 2 ;;
            --vis|--visualizer)
                vis="$2"; shift 2 ;;
            --num-envs|--num_envs)
                num_envs="$2"; shift 2 ;;
            --max-steps|--max_steps)
                max_steps="$2"; shift 2 ;;
            --mode)
                mode="$2"; shift 2 ;;
            --teleop)
                mode="teleop"; shift ;;
            --headless)
                headless=1; shift ;;
            --profile)
                profile=1; shift ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --help|-h)
                usage; exit 0 ;;
            --)
                shift
                extra+=("$@")
                break ;;
            *)
                extra+=("$1"); shift ;;
        esac
    done

    local repo_root
    repo_root="$(resolve_repo_root "$workspace" "$repo_dir_name")"
    local args=(
        scripts/environments/waterhose/run_robot_demo.py
        --task "$task"
        --mode "$mode"
        --num_envs "$num_envs"
        --max_steps "$max_steps"
        --vis "$vis"
    )
    [[ "$headless" == "1" ]] && args+=(--headless)
    [[ "$profile" == "1" ]] && args+=(--profile)
    args+=("${extra[@]}")

    run_isaaclab_python "$repo_root" "$venv_name" "${args[@]}"
}

cmd_teleop() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local task="$DEFAULT_TASK"
    local vis="kit"
    local num_envs=1
    local sensitivity=1.0
    local teleop_device="spacemouse"
    local use_isaac_teleop=0
    local xr=0
    local cloudxr_env="cloudxrjs"
    local auto_launch_cloudxr=1
    local debug_teleop=0
    local venv_name="$DEFAULT_VENV"
    local extra=()

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --task)
                task="$2"; shift 2 ;;
            --vis|--visualizer)
                vis="$2"; shift 2 ;;
            --num-envs|--num_envs)
                num_envs="$2"; shift 2 ;;
            --sensitivity)
                sensitivity="$2"; shift 2 ;;
            --teleop-device|--teleop_device)
                teleop_device="$2"; shift 2 ;;
            --isaac-teleop)
                use_isaac_teleop=1; shift ;;
            --xr)
                xr=1
                use_isaac_teleop=1
                shift ;;
            --cloudxr-env|--cloudxr_env)
                cloudxr_env="$2"; shift 2 ;;
            --no-auto-launch-cloudxr|--no-auto_launch_cloudxr)
                auto_launch_cloudxr=0; shift ;;
            --debug-teleop|--debug_teleop)
                debug_teleop=1; shift ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --help|-h)
                usage; exit 0 ;;
            --)
                shift
                extra+=("$@")
                break ;;
            *)
                extra+=("$1"); shift ;;
        esac
    done

    local repo_root
    repo_root="$(resolve_repo_root "$workspace" "$repo_dir_name")"
    local args=(
        scripts/environments/teleoperation/teleop_se3_agent.py
        --task "$task"
        --num_envs "$num_envs"
        --visualizer "$vis"
        --sensitivity "$sensitivity"
        --cloudxr_env "$cloudxr_env"
    )

    if [[ "$use_isaac_teleop" != "1" ]]; then
        args+=(--teleop_device "$teleop_device")
    fi
    [[ "$xr" == "1" ]] && args+=(--xr)
    [[ "$auto_launch_cloudxr" != "1" ]] && args+=(--no-auto_launch_cloudxr)
    [[ "$debug_teleop" == "1" ]] && args+=(--debug_teleop)
    args+=("${extra[@]}")

    run_isaaclab_python "$repo_root" "$venv_name" "${args[@]}"
}

cmd_smoke() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local venv_name="$DEFAULT_VENV"
    local task="$DEFAULT_TASK"
    local num_envs=1
    local max_steps=20

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --task)
                task="$2"; shift 2 ;;
            --num-envs|--num_envs)
                num_envs="$2"; shift 2 ;;
            --max-steps|--max_steps)
                max_steps="$2"; shift 2 ;;
            --help|-h)
                usage; exit 0 ;;
            *)
                die "Unknown smoke option: $1" ;;
        esac
    done

    cmd_demo \
        --workspace "$workspace" \
        --repo-dir-name "$repo_dir_name" \
        --venv "$venv_name" \
        --task "$task" \
        --vis none \
        --headless \
        --profile \
        --num-envs "$num_envs" \
        --max-steps "$max_steps"
}

cmd_profile() {
    cmd_demo --vis none --headless --profile "$@"
}

cmd_mimic_smoke() {
    cmd_smoke --task "$DEFAULT_MIMIC_TASK" "$@"
}

cmd_python() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local venv_name="$DEFAULT_VENV"

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --help|-h)
                usage; exit 0 ;;
            *)
                break ;;
        esac
    done

    (($#)) || die "python command requires a script or module arguments."
    local repo_root
    repo_root="$(resolve_repo_root "$workspace" "$repo_dir_name")"
    run_isaaclab_python "$repo_root" "$venv_name" "$@"
}

cmd_shell() {
    local workspace="$DEFAULT_WORKSPACE"
    local repo_dir_name="$DEFAULT_REPO_DIR_NAME"
    local venv_name="$DEFAULT_VENV"

    while (($#)); do
        case "$1" in
            --workspace)
                workspace="$(abs_path "$2")"; shift 2 ;;
            --repo-dir-name)
                repo_dir_name="$2"; shift 2 ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --help|-h)
                usage; exit 0 ;;
            *)
                break ;;
        esac
    done

    local repo_root
    repo_root="$(resolve_repo_root "$workspace" "$repo_dir_name")"
    activate_runtime_env "$repo_root" "$venv_name"
    cd "$repo_root"
    log "Activated $(venv_path "$repo_root" "$venv_name"). Type exit to leave the shell."
    exec "${SHELL:-/bin/bash}" "$@"
}

main() {
    local command="${1:-help}"
    [[ $# -gt 0 ]] && shift || true

    case "$command" in
        setup)
            cmd_setup "$@" ;;
        demo)
            cmd_demo "$@" ;;
        teleop)
            cmd_teleop "$@" ;;
        smoke)
            cmd_smoke "$@" ;;
        profile)
            cmd_profile "$@" ;;
        mimic-smoke)
            cmd_mimic_smoke "$@" ;;
        python)
            cmd_python "$@" ;;
        shell)
            cmd_shell "$@" ;;
        help|--help|-h)
            usage ;;
        *)
            die "Unknown command: ${command}. Run ./waterhose.sh help." ;;
    esac
}

main "$@"

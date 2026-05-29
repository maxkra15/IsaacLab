#!/usr/bin/env bash

# One-command setup and run helper for the IsaacLab waterhose demo.
#
# Typical fresh-machine setup:
#   ./waterhose.sh setup --accept-eula --assets-tar /path/to/waterhose_demo_assets.tar.gz
#
# Common run commands after setup:
#   ./waterhose.sh demo --vis kit
#   ./waterhose.sh demo --vis newton --num-envs 4
#   ./waterhose.sh teleop --teleop-device spacemouse --vis kit
#   ./waterhose.sh teleop --xr --cloudxr-env avp --vis kit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

DEFAULT_TASK="Isaac-Waterhose-Robot-Demo-v0"
DEFAULT_MIMIC_TASK="Isaac-Waterhose-Robot-Demo-Mimic-v0"
DEFAULT_ISAACSIM_URL="https://github.com/isaac-sim/IsaacSim.git"
DEFAULT_ISAACSIM_REF="develop"
DEFAULT_ISAACSIM_SRC="${ROOT}/_isaac_sim_src"
DEFAULT_VENV=".venv"
DEFAULT_ASSETS_TAR="${ROOT}/waterhose_demo_assets.tar.gz"

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
  ./waterhose.sh demo [options] [-- extra runner args]
  ./waterhose.sh teleop [options] [-- extra teleop args]
  ./waterhose.sh smoke [options]
  ./waterhose.sh profile [options]
  ./waterhose.sh mimic-smoke [options]
  ./waterhose.sh python <script.py> [args...]
  ./waterhose.sh shell

Setup:
  Clones Isaac Sim from the develop branch, builds the release configuration,
  links it into this repo as _isaac_sim, creates the uv venv, installs IsaacLab
  packages through ./isaaclab.sh, and unpacks the waterhose assets.

  Required on a fresh machine:
    --accept-eula
        Accept the Isaac Sim additional software and materials terms.
    --assets-tar FILE
        Path to waterhose_demo_assets.tar.gz if it is not in the repo root.

Setup options:
  --isaacsim-dir DIR       Isaac Sim source checkout. Default: ./_isaac_sim_src
  --isaacsim-url URL       Isaac Sim repository URL.
  --isaacsim-ref REF       Isaac Sim branch/tag to check out. Default: develop
  --venv DIR               uv virtual environment directory. Default: .venv
  --assets-tar FILE        Waterhose asset archive.
  --jobs N                 Pass -j N to the Isaac Sim build.
  --build-arg ARG          Add one argument to Isaac Sim build.sh.
  --all-configs            Build Isaac Sim default configs instead of release only.
  --accept-eula            Non-interactively accept the Isaac Sim terms.
  --skip-host-deps         Do not install apt packages.
  --skip-gcc-alternatives  Do not set gcc/g++ alternatives to version 11.
  --skip-isaacsim-clone    Reuse an existing Isaac Sim checkout.
  --skip-isaacsim-build    Reuse an existing Isaac Sim build.
  --skip-lfs               Do not run git lfs install/pull.
  --skip-assets            Do not unpack waterhose assets.
  --skip-smoke             Do not run the post-install headless smoke check.

Demo options:
  --task TASK              Task name. Default: Isaac-Waterhose-Robot-Demo-v0
  --vis VALUE              kit, newton, kit,newton, or none. Default: kit
  --num-envs N            Number of environments. Default: 1
  --max-steps N           Demo step limit. Default: 2000
  --headless              Pass --headless to IsaacLab.
  --profile               Print timing from the runner.
  --teleop                Use the demo runner's built-in SpaceMouse teleop mode.
  --venv DIR              uv virtual environment directory. Default: .venv

Teleop options:
  Uses scripts/environments/teleoperation/teleop_se3_agent.py.

  --teleop-device NAME    keyboard, spacemouse, gamepad. Default: spacemouse
  --isaac-teleop          Use env_cfg.isaac_teleop instead of the legacy device path.
  --xr                    Use IsaacTeleop/OpenXR mode. Implies --isaac-teleop.
  --cloudxr-env VALUE     cloudxrjs, avp, none, or a .env path. Default: cloudxrjs
  --no-auto-launch-cloudxr
                          Do not auto-launch the CloudXR runtime.
  --task TASK             Task name. Default: Isaac-Waterhose-Robot-Demo-v0
  --vis VALUE             kit or kit,newton. Default: kit
  --num-envs N            Number of environments. Default: 1
  --sensitivity VALUE     Teleop sensitivity. Default: 1.0
  --debug-teleop          Print periodic teleop diagnostics.
  --venv DIR              uv virtual environment directory. Default: .venv

Examples:
  ./waterhose.sh setup --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz
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

venv_path() {
    local venv_name="$1"
    if [[ "$venv_name" = /* ]]; then
        printf '%s\n' "$venv_name"
    else
        printf '%s/%s\n' "$ROOT" "$venv_name"
    fi
}

sudo_cmd() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install host packages. Re-run with --skip-host-deps to manage them manually."
        sudo "$@"
    fi
}

require_linux() {
    [[ "$(uname -s)" == "Linux" ]] || die "This helper currently supports the Linux Isaac Sim source build path only."
}

ensure_uv() {
    export PATH="${HOME}/.local/bin:${PATH}"
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv."
    log "Installing uv into ${HOME}/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uv >/dev/null 2>&1 || die "uv install completed, but uv is still not on PATH."
}

install_host_deps() {
    require_linux
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get was not found. Skipping host package installation."
        return
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

configure_gcc_11() {
    require_linux
    if ! command -v gcc-11 >/dev/null 2>&1 || ! command -v g++-11 >/dev/null 2>&1; then
        warn "gcc-11/g++-11 were not found. Isaac Sim build may fail the compiler version check."
        return
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

ensure_clean_git_checkout() {
    local repo_dir="$1"
    [[ -d "${repo_dir}/.git" ]] || return
    if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
        die "Existing Isaac Sim checkout has local changes: ${repo_dir}. Commit/stash them or pass a different --isaacsim-dir."
    fi
}

prepare_isaacsim_checkout() {
    local isaacsim_dir="$1"
    local isaacsim_url="$2"
    local isaacsim_ref="$3"
    local skip_clone="$4"
    local skip_lfs="$5"

    if [[ "$skip_clone" == "1" ]]; then
        [[ -d "${isaacsim_dir}/.git" ]] || die "--skip-isaacsim-clone requires an existing git checkout at ${isaacsim_dir}"
    elif [[ ! -d "${isaacsim_dir}/.git" ]]; then
        mkdir -p "$(dirname "$isaacsim_dir")"
        log "Cloning Isaac Sim ${isaacsim_ref} into ${isaacsim_dir}"
        GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$isaacsim_ref" "$isaacsim_url" "$isaacsim_dir"
    else
        ensure_clean_git_checkout "$isaacsim_dir"
        log "Updating existing Isaac Sim checkout at ${isaacsim_dir}"
        run git -C "$isaacsim_dir" fetch --all --tags --prune
        if git -C "$isaacsim_dir" show-ref --verify --quiet "refs/remotes/origin/${isaacsim_ref}"; then
            run git -C "$isaacsim_dir" checkout -B "$isaacsim_ref" "origin/${isaacsim_ref}"
        else
            run git -C "$isaacsim_dir" checkout "$isaacsim_ref"
        fi
        if [[ "$(git -C "$isaacsim_dir" rev-parse --abbrev-ref HEAD)" == "$isaacsim_ref" ]]; then
            run git -C "$isaacsim_dir" pull --ff-only
        fi
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
        warn "Isaac Sim will ask for EULA acceptance during build. Use --accept-eula for non-interactive setup."
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
    local isaacsim_dir="$1"
    local build_dir
    build_dir="$(isaacsim_build_dir "$isaacsim_dir")"

    [[ -d "$build_dir" ]] || die "Expected Isaac Sim build directory does not exist: ${build_dir}"
    [[ -f "${build_dir}/python.sh" ]] || die "Isaac Sim python.sh not found in ${build_dir}"
    [[ -f "${build_dir}/setup_conda_env.sh" ]] || die "Isaac Sim setup_conda_env.sh not found in ${build_dir}"

    ln -sfn "$build_dir" "${ROOT}/_isaac_sim"
    log "Linked ${ROOT}/_isaac_sim -> ${build_dir}"
}

setup_uv_env_and_install_isaaclab() {
    local venv_name="$1"
    local venv_dir
    venv_dir="$(venv_path "$venv_name")"

    ensure_uv
    [[ -e "${ROOT}/_isaac_sim" ]] || die "_isaac_sim is missing. Build or link Isaac Sim before creating the env."

    run "${ROOT}/isaaclab.sh" --uv "$venv_name"
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
    run "${ROOT}/isaaclab.sh" -i all
}

extract_assets() {
    local assets_tar="$1"
    local target_dir="${ROOT}/source/isaaclab_assets/data"
    local expected_dir="${target_dir}/WaterhoseDemo"

    if [[ -d "$expected_dir" ]]; then
        log "Waterhose assets already present at ${expected_dir}"
        return
    fi
    [[ -f "$assets_tar" ]] || die "Asset archive not found: ${assets_tar}. Copy waterhose_demo_assets.tar.gz into the repo root or pass --assets-tar FILE."

    mkdir -p "$target_dir"
    log "Unpacking waterhose assets into ${target_dir}"
    tar -xzf "$assets_tar" -C "$target_dir"
    [[ -d "$expected_dir" ]] || die "Asset archive did not create ${expected_dir}"
}

activate_runtime_env() {
    local venv_name="$1"
    local venv_dir
    venv_dir="$(venv_path "$venv_name")"
    local activate="${venv_dir}/bin/activate"
    [[ -f "$activate" ]] || die "Virtual environment not found: ${venv_dir}. Run ./waterhose.sh setup first."
    # shellcheck disable=SC1090
    source "$activate"
    if [[ -f "${ROOT}/_isaac_sim/setup_conda_env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${ROOT}/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    fi
    export PYTHONPATH="${ROOT}/source/isaaclab:${PYTHONPATH:-}"
}

run_isaaclab_python() {
    local venv_name="$1"
    shift
    activate_runtime_env "$venv_name"
    (cd "$ROOT" && exec "${ROOT}/isaaclab.sh" -p "$@")
}

cmd_setup() {
    local isaacsim_dir="$DEFAULT_ISAACSIM_SRC"
    local isaacsim_url="$DEFAULT_ISAACSIM_URL"
    local isaacsim_ref="$DEFAULT_ISAACSIM_REF"
    local venv_name="$DEFAULT_VENV"
    local assets_tar="$DEFAULT_ASSETS_TAR"
    local accept_eula=0
    local skip_host_deps=0
    local skip_gcc_alternatives=0
    local skip_clone=0
    local skip_build=0
    local skip_lfs=0
    local skip_assets=0
    local skip_smoke=0
    local release_only=1
    local build_args=()

    while (($#)); do
        case "$1" in
            --isaacsim-dir)
                isaacsim_dir="$(abs_path "$2")"; shift 2 ;;
            --isaacsim-url)
                isaacsim_url="$2"; shift 2 ;;
            --isaacsim-ref)
                isaacsim_ref="$2"; shift 2 ;;
            --venv)
                venv_name="$2"; shift 2 ;;
            --assets-tar)
                assets_tar="$(abs_path "$2")"; shift 2 ;;
            --jobs|-j)
                build_args+=("-j" "$2"); shift 2 ;;
            --build-arg)
                build_args+=("$2"); shift 2 ;;
            --all-configs)
                release_only=0; shift ;;
            --accept-eula)
                accept_eula=1; shift ;;
            --skip-host-deps)
                skip_host_deps=1; shift ;;
            --skip-gcc-alternatives)
                skip_gcc_alternatives=1; shift ;;
            --skip-isaacsim-clone)
                skip_clone=1; shift ;;
            --skip-isaacsim-build)
                skip_build=1; shift ;;
            --skip-lfs)
                skip_lfs=1; shift ;;
            --skip-assets)
                skip_assets=1; shift ;;
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
    if [[ "$skip_host_deps" != "1" ]]; then
        install_host_deps
        if [[ "$skip_gcc_alternatives" != "1" ]]; then
            configure_gcc_11
        fi
        ensure_uv
    else
        ensure_uv
    fi

    prepare_isaacsim_checkout "$isaacsim_dir" "$isaacsim_url" "$isaacsim_ref" "$skip_clone" "$skip_lfs"
    accept_isaacsim_eula_if_requested "$isaacsim_dir" "$accept_eula"

    if [[ "$release_only" == "1" ]]; then
        build_args=("-r" "${build_args[@]}")
    fi
    if [[ "$skip_build" != "1" ]]; then
        build_isaacsim "$isaacsim_dir" "${build_args[@]}"
    fi
    link_isaacsim_build "$isaacsim_dir"

    setup_uv_env_and_install_isaaclab "$venv_name"

    if [[ "$skip_assets" != "1" ]]; then
        extract_assets "$assets_tar"
    fi

    if [[ "$skip_smoke" != "1" ]]; then
        cmd_smoke --venv "$venv_name"
    fi

    log "Setup complete."
}

cmd_demo() {
    local task="$DEFAULT_TASK"
    local vis="kit"
    local num_envs=1
    local max_steps=2000
    local mode="scripted"
    local venv_name="${WATERHOSE_VENV:-$DEFAULT_VENV}"
    local headless=0
    local profile=0
    local extra=()

    while (($#)); do
        case "$1" in
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

    run_isaaclab_python "$venv_name" "${args[@]}"
}

cmd_teleop() {
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
    local venv_name="${WATERHOSE_VENV:-$DEFAULT_VENV}"
    local extra=()

    while (($#)); do
        case "$1" in
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

    run_isaaclab_python "$venv_name" "${args[@]}"
}

cmd_smoke() {
    local venv_name="${WATERHOSE_VENV:-$DEFAULT_VENV}"
    local task="$DEFAULT_TASK"
    local num_envs=1
    local max_steps=20

    while (($#)); do
        case "$1" in
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

    cmd_demo --venv "$venv_name" --task "$task" --vis none --headless --profile --num-envs "$num_envs" --max-steps "$max_steps"
}

cmd_profile() {
    cmd_demo --vis none --headless --profile "$@"
}

cmd_mimic_smoke() {
    local args=("--task" "$DEFAULT_MIMIC_TASK")
    args+=("$@")
    cmd_smoke "${args[@]}"
}

cmd_python() {
    local venv_name="${WATERHOSE_VENV:-$DEFAULT_VENV}"
    if [[ "${1:-}" == "--venv" ]]; then
        venv_name="$2"
        shift 2
    fi
    (($#)) || die "python command requires a script or module arguments."
    run_isaaclab_python "$venv_name" "$@"
}

cmd_shell() {
    local venv_name="${WATERHOSE_VENV:-$DEFAULT_VENV}"
    if [[ "${1:-}" == "--venv" ]]; then
        venv_name="$2"
        shift 2
    fi
    activate_runtime_env "$venv_name"
    cd "$ROOT"
    log "Activated $(venv_path "$venv_name"). Type exit to leave the shell."
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

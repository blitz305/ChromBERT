"""Run a benchmark container directly on the current GPU server."""

import csv
import fcntl
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from archives import resolve_reference_for_launch
from variants import get_variant


# Idle framebuffer usage observed on this project's A100 server.
IDLE_MEMORY_MIB = 4


def capture(*command):
    return subprocess.check_output(command, universal_newlines=True).strip()


def active_workspace_container(workspace):
    containers = capture("docker", "ps", "-q", "--filter", "name=chrombert-official_").splitlines()
    if not containers:
        return None
    for container in json.loads(capture("docker", "inspect", *containers)):
        name = container["Name"].lstrip("/")
        if not name.startswith("chrombert-official_"):
            continue
        sources = (Path(mount["Source"]) for mount in container["Mounts"] if mount["Type"] == "bind")
        if any(source == workspace or workspace in source.parents for source in sources):
            return name
    return None


def available_gpu(index, workspace=None):
    if index < 0:
        raise ValueError("GPU index must be nonnegative.")
    devices = capture("nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits")
    state = {}
    for row in csv.reader(devices.splitlines(), skipinitialspace=True):
        card_index, uuid, memory = row
        state[int(card_index)] = (uuid, int(memory))
    if index not in state:
        raise ValueError(f"GPU {index} is unavailable.")
    uuid, memory = state[index]
    active = capture("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
    for device, pid in csv.reader(active.splitlines(), skipinitialspace=True):
        if device == uuid:
            raise ValueError(f"GPU {index} has active compute process {pid}.")
        if workspace is not None and workspace_process(workspace, int(pid)):
            raise ValueError(f"Project GPU process {pid} is still running on {device}.")
    if memory > IDLE_MEMORY_MIB:
        raise ValueError(f"GPU {index} has {memory} MiB in use and is not idle.")
    return uuid


def workspace_process(workspace, pid):
    proc = Path("/proc") / str(pid)
    try:
        cwd = (proc / "cwd").resolve(strict=True)
        if cwd == workspace or workspace in cwd.parents:
            return True
    except OSError:
        pass
    try:
        cmdline = (proc / "cmdline").read_bytes()
        return os.fsencode(str(workspace) + "/") in cmdline
    except OSError:
        return False


def prepare_output(directory):
    options = json.loads(capture("docker", "info", "--format", "{{json .SecurityOptions}}"))
    if not any("name=userns" in option for option in options or []):
        return
    root = Path(capture("docker", "info", "--format", "{{.DockerRootDir}}"))
    ids = re.fullmatch(r"(\d+)\.(\d+)", root.name)
    if ids is None:
        raise ValueError(f"Cannot determine the Docker user namespace IDs from {root}.")
    access = f"u:{ids[1]}:rwx,d:u:{os.getuid()}:rwx"
    subprocess.run(["setfacl", "-m", access, str(directory)], check=True)


def check_in_container(code, output):
    image = capture("docker", "inspect", "--format", "{{.Image}}", "epimodel_container")
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--network", "none",
        "--mount", f"type=bind,src={code},dst=/benchmark,readonly",
        "--mount", f"type=bind,src={output.resolve()},dst=/result,readonly",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python", image,
        "/benchmark/benchmark.py", "check", "/result",
    ], check=True)


def launch(task, mode, gpu_index, workspace, build_bundle, assets=None, reference=None, variant=None):
    workspace = workspace.resolve()
    assets = assets.resolve() if assets else workspace / "assets/hg38_1kb"
    if not assets.is_dir():
        raise ValueError(f"Missing benchmark assets directory: {assets}")
    variant = variant or get_variant("original")
    reference = resolve_reference_for_launch(reference, mode, variant)
    bundles = workspace / "bundles"
    runs = workspace / "runs/benchmark"
    bundles.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    with (runs / ".official-benchmark-queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A benchmark queue is already using the workspace GPU slot.") from None
        active = active_workspace_container(workspace)
        if active:
            raise ValueError(f"Benchmark container {active} is already running in this workspace.")
        gpu = available_gpu(gpu_index, workspace)
        image, shm_size = capture("docker", "inspect", "--format", "{{.Image}} {{.HostConfig.ShmSize}}",
                                  "epimodel_container").split()
        prefix = f"official_{task}"
        if mode in {"profile", "smoke-profile"}:
            prefix += "_profile"
        if mode == "profile-only":
            prefix += "_profile-only"
        prefix += variant.archive_suffix
        if mode in {"smoke", "smoke-profile"}:
            prefix += "_smoke"
        bundle = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(bundles)))
        bundle.chmod(0o755)
        build_bundle(task, bundle)
        docker = [
            "docker", "run", "--pull=never", "--network", "none",
            "--gpus", f"device={gpu}", "--shm-size", shm_size,
            "--mount", f"type=bind,src={bundle},dst=/benchmark,readonly",
            "--mount", f"type=bind,src={assets},dst=/assets,readonly",
            "-e", f"CUDA_VISIBLE_DEVICES={gpu}", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "PYTORCH_CUDA_ALLOC_CONF=backend:native",
        ]
        runtime = ["env", "-u", "PYTORCH_NO_CUDA_MEMORY_CACHING", "python", "-u", "/benchmark/benchmark.py"]
        paths = ["--bundle", "/benchmark", "--assets", "/assets"]
        if mode == "preflight":
            if available_gpu(gpu_index, workspace) != gpu:
                raise ValueError("GPU identity changed before launch.")
            subprocess.run([*docker, "--rm", "--entrypoint", runtime[0], image, *runtime[1:], "preflight", *paths],
                           check=True)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(tempfile.mkdtemp(prefix=f"{prefix}_{timestamp}_", dir=str(runs)))
        output.chmod(0o750)
        prepare_output(output)
        container = f"chrombert-{output.name}"
        command = [*runtime, "run", *paths, "--output", "/output/result"]
        if mode in {"profile", "profile-only", "smoke-profile"}:
            command.append("--profile")
        if mode in {"smoke", "smoke-profile"}:
            command.append("--smoke")
        if reference is not None:
            docker.extend(["--mount", f"type=bind,src={reference},dst=/reference,readonly"])
            command.extend(["--from-run", "/reference"])
            (output / "reference_path").write_text(str(reference) + "\n", encoding="utf-8")
        command.extend(["--variant", variant.name])
        worker = (
            'set -o pipefail\n'
            'code=0\n'
            '"$@" 2>&1 | tee /output/run.log || code=$?\n'
            'printf "%s\\n" "$code" > /output/exit_code\n'
            'exit "$code"\n'
        )
        (output / "container_name").write_text(container + "\n", encoding="utf-8")
        (output / "bundle_path").write_text(str(bundle) + "\n", encoding="utf-8")
        if available_gpu(gpu_index, workspace) != gpu:
            raise ValueError("GPU identity changed before launch.")
        subprocess.run([
            *docker, "-d", "--name", container,
            "--mount", f"type=bind,src={output},dst=/output",
            "--entrypoint", "bash", image, "-c", worker, "benchmark", *command,
        ], check=True, stdout=subprocess.DEVNULL)
        print(f"GPU: {gpu_index} ({gpu})\nLog: {output / 'run.log'}\nResults: {output / 'result'}")

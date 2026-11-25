"""Bundle, run, and verify the ChromBERT benchmarks based on the official tutorials."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from functools import partial
from pathlib import Path

from archives import verify_result
from benchmark_io import BENCHMARK_SCRIPTS, CODE, profiling_enabled, read_json, sha256, write_json
from model_sources import get_source, source_digest, verify_notebook
from variants import add_variant_options, chosen_variant, configured_run, mode_for_worker, validate_mode
from tasks import TASKS, get_task, prepare_splits
from tasks.cistrome import SAMPLE_FILENAME, create_sample
from tasks.eqtl import EQTL_TABLE
from tasks.gep import GEP_DIRECTORY, GEP_FILES
from tasks.starr import STARR_TABLES

REPO = CODE.parent
DEMO_SHA256 = "60f1777bd53c55e14f720f858fdcc545c9ec168edcadf318166da1405126e6ab"


def bundle(task, destination, workspace=None):
    config_path = CODE / "configs" / f"{task}.json"
    config = read_json(config_path)
    profiling_enabled(config)
    destination.mkdir(parents=True, exist_ok=True)
    get_source(config).bundle(config, destination, REPO)
    shutil.copyfile(config_path, destination / "task.json")
    for script in BENCHMARK_SCRIPTS:
        target = destination / script
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CODE / script, target)
    if workspace is None:
        return
    for relative, local in config["bundled_inputs"].items():
        source = workspace / local
        if not source.is_file():
            continue  # The same official file may already be installed on the server.
        expected = config["input_sha256"].get(relative)
        if expected and sha256(source) != expected:
            raise ValueError(f"Bundled file differs from the official input: {source}")
        target = destination / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def extract_gep(archive, destination):
    extract_demo_assets(archive, destination, {f"{GEP_DIRECTORY}/{name}": name for name in GEP_FILES})


def extract_demo_assets(archive, destination, wanted):
    if sha256(archive) != DEMO_SHA256:
        raise ValueError("The archive does not match the official demo.tar.gz SHA-256.")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as directory:
        staging = Path(directory)
        with tarfile.open(archive, "r|gz") as source:
            for member in source:
                path = member.name[2:] if member.name.startswith("./") else member.name
                if path not in wanted:
                    continue
                target = staging / wanted[path]
                if not member.isfile() or target.exists():
                    raise ValueError(f"Unexpected or duplicate official demo member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                if all((staging / name).is_file() for name in wanted.values()):
                    break
        missing = [name for name in wanted.values() if not (staging / name).is_file()]
        if missing:
            raise ValueError(f"Missing members in the official demo archive: {', '.join(missing)}")
        for name in wanted.values():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(destination / name)
            print(f"{sha256(destination / name)}  {destination / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    pack = commands.add_parser("bundle")
    pack.add_argument("task", choices=TASKS)
    pack.add_argument("destination", type=Path)
    pack.add_argument("--workspace", type=Path, help="include cached inputs from this shared workspace")
    local = commands.add_parser("launch", help="start a Docker run on this server")
    local.add_argument("task", choices=TASKS)
    local.add_argument("--mode", choices=("full", "profile", "profile-only", "smoke", "smoke-profile", "preflight"),
                       default="full")
    local.add_argument("--gpu", type=int, required=True)
    local.add_argument("--workspace", type=Path, required=True,
                       help="shared directory for assets, input cache, bundles, and runs")
    local.add_argument("--assets", type=Path)
    local.add_argument("--from-run", type=Path, help="archive with verified results for profiling or saved splits for GELU comparison")
    add_variant_options(local)
    assets = commands.add_parser("extract-gep")
    assets.add_argument("archive", type=Path)
    assets.add_argument("--destination", type=Path, required=True)
    demo = commands.add_parser("extract-demo")
    demo.add_argument("archive", type=Path)
    demo.add_argument("--destination", type=Path, required=True)
    sample = commands.add_parser("sample-cistrome", help="generate the fixed MAZ × K562 sample on CPU")
    sample.add_argument("full_split", type=Path)
    sample.add_argument("--destination", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--bundle", type=Path, required=True)
    audit.add_argument("--assets", type=Path, required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--assets", type=Path, required=True)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--profile", action="store_true")
            command.add_argument("--profile-gep", action="store_true")
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--from-run", type=Path)
            add_variant_options(command)
    check = commands.add_parser("check")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "bundle":
            bundle(args.task, args.destination, args.workspace)
        elif args.command == "launch":
            from launcher import launch

            variant = chosen_variant(args)
            validate_mode(args.task, get_task(args.task), args.mode, variant, args.from_run)
            workspace = args.workspace.resolve()
            build_bundle = partial(bundle, workspace=workspace)
            launch(args.task, args.mode, args.gpu, workspace, build_bundle, args.assets, args.from_run, variant)
        elif args.command == "extract-gep":
            extract_gep(args.archive, args.destination)
        elif args.command == "extract-demo":
            wanted = (EQTL_TABLE, *STARR_TABLES.values())
            extract_demo_assets(args.archive, args.destination, {path: path for path in wanted})
        elif args.command == "sample-cistrome":
            destination = args.destination.resolve()
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / SAMPLE_FILENAME
            manifest = output.with_suffix(".json")
            if output.exists() or manifest.exists():
                raise ValueError(f"A fixed sample already exists in {destination}.")
            with tempfile.TemporaryDirectory(dir=str(destination)) as staging:
                staged = Path(staging) / SAMPLE_FILENAME
                data = create_sample(args.full_split, staged)
                data.update(
                    source_sha256=sha256(args.full_split), indices_sha256=sha256(staged),
                    generator_python=sys.version.split()[0], index_file=SAMPLE_FILENAME,
                )
                write_json(staged.with_suffix(".json"), data)
                staged.rename(output)
                staged.with_suffix(".json").rename(manifest)
            print(f"Sample: {output}\nManifest: {manifest}")
        elif args.command == "check":
            output = args.output
            if not (output / "config.json").is_file() and (output / "result" / "config.json").is_file():
                output = output / "result"
            if not (output / "config.json").is_file():
                raise ValueError(f"No config.json found in {output}.")
            if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "pandas", "sklearn")):
                from launcher import check_in_container

                check_in_container(CODE, output)
            else:
                print(json.dumps(verify_result(output), indent=2))
        else:
            from runner import execute, input_files, preflight

            directory = args.bundle.resolve()
            config = read_json(directory / "task.json")
            profiling_enabled(config)
            get_source(config).verify(config, directory)
            protocol = get_task(config["task"])
            if args.command == "audit":
                if not protocol.AUDIT_SPLITS:
                    raise ValueError("audit is for eQTL and STARR.")
                with tempfile.TemporaryDirectory() as staging:
                    _, manifest = prepare_splits(
                        config, input_files(config, directory, args.assets.resolve()), Path(staging) / "splits",
                    )
                    print(json.dumps(manifest, indent=2))
            elif args.command == "preflight":
                print(json.dumps(preflight(config, directory, args.assets.resolve()), indent=2))
            else:
                if args.profile_gep and config["task"] != "gep":
                    raise ValueError("--profile-gep requires the GEP benchmark.")
                variant = chosen_variant(args)
                if args.profile_gep and variant.requires_baseline:
                    raise ValueError("--profile-gep is for the baseline; use --profile with a comparison reference.")
                mode = mode_for_worker(args, variant)
                reference = args.from_run.resolve() if args.from_run else None
                config = configured_run(config, protocol, mode, variant, reference)
                execute(config, directory, args.assets.resolve(), args.output.resolve(), verify_result, reference)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Benchmark error: {error}\n")


if __name__ == "__main__":
    main()

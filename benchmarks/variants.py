"""Registered variants and their rules for reusing benchmark references."""

from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

from benchmark_io import profiling_enabled


class Variant:
    name = "original"
    archive_suffix = ""
    requires_reference = False
    requires_baseline = False
    match_environment = False
    modes = {"full", "profile", "profile-only", "smoke", "smoke-profile", "preflight"}

    def activate(self):
        return ExitStack()

    def record(self, mode):
        return {"variant": self.name}

    def uses_splits_only(self, config):
        return False

    def uses_fixed_checkpoint_evaluation(self, config):
        return False


class BuiltinGelu(Variant):
    name = "gelu-builtin"
    archive_suffix = "_gelu-builtin"
    requires_reference = True
    requires_baseline = True
    match_environment = True
    modes = {"full", "profile-only"}

    def activate(self):
        import torch

        print("Using PyTorch GELU (tanh) in ChromBERT FFNs", flush=True)
        return patch("chrombert.base.utils.feed_forward.GELU", partial(torch.nn.GELU, approximate="tanh"))

    def record(self, mode):
        # Retain the original metadata for existing archive readers.
        return {
            "variant": self.name,
            "gelu": "builtin",
            "gelu_comparison": "profile_only" if mode == "profile-only" else "end_to_end",
        }

    def uses_splits_only(self, config):
        return config.get("gelu_comparison") == "end_to_end"

    def uses_fixed_checkpoint_evaluation(self, config):
        return not self.uses_splits_only(config)


VARIANTS = {variant.name: variant for variant in (Variant(), BuiltinGelu())}
LEGACY_GELU = {"original": "original", "builtin": "gelu-builtin"}


def get_variant(name):
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark variant: {name}") from None


def variant_from_config(config):
    if "variant" in config:
        return get_variant(config["variant"])
    return get_variant(LEGACY_GELU[config.get("gelu", "original")])


def add_variant_options(parser):
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--variant", choices=VARIANTS, help="registered benchmark variant")
    choices.add_argument("--gelu", choices=LEGACY_GELU, help="alias for the original or gelu-builtin variant")


def chosen_variant(args):
    return get_variant(args.variant or LEGACY_GELU[args.gelu or "original"])


def validate_mode(task, protocol, mode, variant, reference):
    if mode not in variant.modes:
        raise ValueError(f"{variant.name} does not support mode {mode}.")
    if mode.startswith("smoke") and not protocol.SUPPORTS_SMOKE:
        raise ValueError(f"{task} has no smoke mode. Use its normal task configuration.")
    if mode in {"profile", "smoke-profile"} and not protocol.SUPPORTS_PROFILING:
        raise ValueError(f"{task} does not support profiler mode.")
    if (mode == "profile-only" or variant.requires_reference) != (reference is not None):
        raise ValueError("--from-run is required for profile-only and comparison variants.")


def mode_for_worker(args, variant):
    if args.from_run and args.smoke:
        raise ValueError("--from-run uses the saved full splits and cannot be combined with --smoke.")
    profile = args.profile or args.profile_gep
    if args.from_run and (profile or not variant.requires_reference):
        return "profile-only"
    if args.smoke:
        return "smoke-profile" if profile else "smoke"
    return "profile" if profile else "full"


def configured_run(config, protocol, mode, variant, reference):
    validate_mode(config["task"], protocol, mode, variant, reference)
    if variant.requires_baseline and profiling_enabled(config):
        raise ValueError(f"{variant.name} requires an unprofiled base config.")
    result = {**config, **variant.record(mode)}
    if mode in {"profile", "profile-only", "smoke-profile"}:
        result["profile"] = True
    if mode == "profile-only":
        result["profile_only"] = True
    if mode in {"smoke", "smoke-profile"}:
        result["smoke"] = True
    return result

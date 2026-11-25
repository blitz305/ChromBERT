"""Model-source adapters and provenance for benchmark bundles."""

import hashlib
import shutil
import subprocess
from pathlib import Path

from benchmark_io import sha256

REFERENCE_REVISION = "10589002597a9ef06613ed41b6ee83e8dd565c25"
REFERENCE_TREE_SHA256 = "1cd740c975e8a2b673a52ce71b51c4f9766925681686025c2b961beba3ddb376"


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def verify_notebook(config, notebook):
    if sha256(notebook) != config["notebook_sha256"]:
        raise ValueError(f"Notebook differs from the pinned official revision: {notebook}")


class OfficialSource:
    """Reproduce a pinned official ChromBERT tutorial."""

    reference_fields = ("official_revision", "notebook_sha256")

    def bundle(self, config, destination, repo):
        revision = config["official_revision"]
        base = ("git",)
        destination.mkdir(parents=True, exist_ok=True)
        notebook = destination / "official.ipynb"
        has_revision = subprocess.run(
            (*base, "cat-file", "-e", f"{revision}^{{commit}}"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=repo,
        ).returncode == 0
        if has_revision:
            notebook.write_bytes(subprocess.check_output((*base, "show", f"{revision}:{config['notebook']}"), cwd=repo))
            with subprocess.Popen((*base, "archive", revision, "chrombert"), stdout=subprocess.PIPE, cwd=repo) as process:
                subprocess.run(("tar", "-xf", "-", "-C", str(destination)), stdin=process.stdout, check=True)
                if process.wait():
                    raise ValueError("Could not export the pinned official ChromBERT source.")
        else:
            reference = repo / "chrombert" / "reference" / "ChromBERT"
            source = reference / "chrombert"
            if revision != REFERENCE_REVISION or source_digest(source) != REFERENCE_TREE_SHA256:
                raise ValueError("No Git revision or matching official reference snapshot was found.")
            shutil.copyfile(reference / config["notebook"], notebook)
            shutil.copytree(source, destination / "chrombert", ignore=shutil.ignore_patterns("__pycache__"))
        verify_notebook(config, notebook)

    def verify(self, config, directory):
        verify_notebook(config, directory / "official.ipynb")

    def archive(self, config, bundle, output):
        shutil.copyfile(bundle / "official.ipynb", output / "official.ipynb")

    def artifacts(self, config):
        return {"official.ipynb"}

    def metadata(self, config, bundle):
        import chrombert

        source = Path(chrombert.__file__).resolve()
        if source.parent != (bundle / "chrombert").resolve():
            raise RuntimeError(f"Unexpected ChromBERT source: {source}")
        return {**{key: config[key] for key in self.reference_fields}, "chrombert_source": str(source)}


MODEL_SOURCES = {"official": OfficialSource()}


def get_source(config):
    name = config.get("model_source", "official")
    try:
        return MODEL_SOURCES[name]
    except KeyError:
        raise ValueError(f"Unknown model source: {name}") from None

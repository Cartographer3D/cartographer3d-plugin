from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from typing_extensions import override


def _run_git_command(cmd: str) -> str:
    prog = shlex.split(cmd)
    process = subprocess.Popen(prog, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret, _ = process.communicate()
    retcode = process.wait()
    if retcode == 0:
        return ret.strip().decode()
    return ""


def get_commit_sha(root: str) -> str:
    sha = _run_git_command(f"git -C {root} rev-parse HEAD")
    return sha or os.getenv("COMMIT_SHA", "")


def retrieve_git_version(root: str) -> str:
    version = _run_git_command(f"git -C {root} describe --always --tags --long --dirty")
    return version or os.getenv("GIT_VERSION", "")


def _is_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true"


class ReleaseInfo(TypedDict):
    project_name: str
    package_name: str
    urls: dict[str, str]
    package_version: str
    git_version: str
    commit_sha: str
    build_time: str
    system_dependencies: dict[str, list[str]]


class Project(TypedDict):
    name: str
    urls: dict[str, str]


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    @override
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        project: Project | None = self.metadata.config.get("project")  # pyright: ignore[reportUnknownMemberType]
        if project is None:
            msg = "Project not found in metadata"
            raise ValueError(msg)

        build_time = datetime.now(timezone.utc)
        urls = project.get("urls", {})

        ref_name = os.getenv("GITHUB_REF_NAME")
        repository = os.getenv("GITHUB_REPOSITORY")
        if ref_name is not None and repository is not None:
            urls["changelog"] = f"https://github.com/{repository}/releases/tag/{ref_name}"

        git_version = retrieve_git_version(self.root)
        commit_sha = get_commit_sha(self.root)

        if _is_ci() and (not git_version or not commit_sha):
            msg = f"CI build is missing git metadata: git_version={git_version!r}, commit_sha={commit_sha!r}"
            raise RuntimeError(msg)

        data = ReleaseInfo(
            project_name=project.get("name"),
            package_name=self.metadata.name,  # pyright: ignore[reportUnknownMemberType]
            urls=urls,
            package_version=self.metadata.version,  # pyright: ignore[reportUnknownMemberType]
            git_version=git_version,
            commit_sha=commit_sha,
            build_time=datetime.isoformat(build_time, timespec="seconds"),
            system_dependencies={},
        )

        out_file = Path(self.root, "release_info")
        with open(out_file, "w") as f:
            json.dump(data, f)

        build_data["extra_metadata"][str(out_file)] = "release_info"

    @override
    def clean(self, versions: list[str]) -> None:
        out_file = Path(self.root, "release_info")
        out_file.unlink(missing_ok=True)

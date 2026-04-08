"""Tests for OS utilities."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab.utils.os import get_wandb_checkpoint_path


class _FakeWandbFile:
  def __init__(self, name: str):
    self.name = name


class _FakeDownloadable:
  def __init__(self, name: str, should_fail: bool = False):
    self.name = name
    self.should_fail = should_fail

  def download(self, directory: str, replace: bool = True):
    del replace
    if self.should_fail:
      raise RuntimeError(f"download failed for {self.name}")
    Path(directory, self.name).write_text("checkpoint")


class _FakeRun:
  def __init__(
    self,
    file_names: list[str],
    files_error: Exception | None = None,
    direct_files: set[str] | None = None,
    summary: dict | None = None,
    failing_downloads: set[str] | None = None,
  ):
    self._file_names = file_names
    self._files_error = files_error
    self._direct_files = direct_files or set(file_names)
    self.summary = summary or {}
    self._failing_downloads = failing_downloads or set()

  def files(self):
    if self._files_error is not None:
      raise self._files_error
    return [_FakeWandbFile(name) for name in self._file_names]

  def file(self, name: str):
    if name in self._direct_files:
      return _FakeDownloadable(name, should_fail=name in self._failing_downloads)
    return None


class _FakeApi:
  def __init__(self, run: _FakeRun):
    self._run = run

  def run(self, run_path: str):
    del run_path
    return self._run


def test_get_wandb_checkpoint_path_falls_back_to_last_pt_when_files_list_fails(
  tmp_path, monkeypatch
):
  run = _FakeRun(
    file_names=[],
    files_error=TypeError("'NoneType' object is not subscriptable"),
    direct_files={"last.pt"},
  )
  api = _FakeApi(run)
  fake_wandb = SimpleNamespace(Api=lambda: api)
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  checkpoint_path, was_cached = get_wandb_checkpoint_path(
    tmp_path, Path("entity/project/runs/run456")
  )

  assert checkpoint_path == tmp_path / "wandb_checkpoints" / "run456" / "last.pt"
  assert checkpoint_path.exists()
  assert not was_cached


def test_get_wandb_checkpoint_path_uses_summary_step_model_when_listing_fails(
  tmp_path, monkeypatch
):
  run = _FakeRun(
    file_names=[],
    files_error=TypeError("'NoneType' object is not subscriptable"),
    direct_files={"model_44999.pt"},
    summary={"_step": 44999},
  )
  api = _FakeApi(run)
  fake_wandb = SimpleNamespace(Api=lambda: api)
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  checkpoint_path, was_cached = get_wandb_checkpoint_path(
    tmp_path, Path("entity/project/runs/run44999")
  )

  assert checkpoint_path == (
    tmp_path / "wandb_checkpoints" / "run44999" / "model_44999.pt"
  )
  assert checkpoint_path.exists()
  assert not was_cached


def test_get_wandb_checkpoint_path_retries_candidates_after_download_failure(
  tmp_path, monkeypatch
):
  run = _FakeRun(
    file_names=[],
    files_error=TypeError("'NoneType' object is not subscriptable"),
    direct_files={"last.pt", "model.pt"},
    failing_downloads={"last.pt"},
  )
  api = _FakeApi(run)
  fake_wandb = SimpleNamespace(Api=lambda: api)
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  checkpoint_path, was_cached = get_wandb_checkpoint_path(
    tmp_path, Path("entity/project/runs/runretry")
  )

  assert checkpoint_path == tmp_path / "wandb_checkpoints" / "runretry" / "model.pt"
  assert checkpoint_path.exists()
  assert not was_cached


def test_get_wandb_checkpoint_path_raises_when_no_checkpoint_can_be_resolved(
  tmp_path, monkeypatch
):
  run = _FakeRun(
    file_names=[],
    files_error=TypeError("'NoneType' object is not subscriptable"),
    direct_files=set(),
  )
  api = _FakeApi(run)
  fake_wandb = SimpleNamespace(Api=lambda: api)
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  with pytest.raises(ValueError, match="Could not download any checkpoint"):
    get_wandb_checkpoint_path(tmp_path, Path("entity/project/runs/run789"))

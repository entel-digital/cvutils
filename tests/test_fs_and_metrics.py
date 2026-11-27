import tempfile
from pathlib import Path

from cvutils.pipeline_task.utils.fs import list_files, walk_to_level


def test_list_files_default_valid_exts_and_extensionless():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "README").write_text("docs", encoding="utf-8")
        (root / "image.JPG").write_text("jpg", encoding="utf-8")
        sub = root / "nested"
        sub.mkdir()
        (sub / "frame.png").write_text("png", encoding="utf-8")

        all_files = sorted(Path(p).name for p in list_files(str(root)))
        assert all_files == ["README", "frame.png", "image.JPG"]

        img_files = sorted(Path(p).name for p in list_files(str(root), valid_exts=(".jpg", ".png")))
        assert img_files == ["frame.png", "image.JPG"]

        top_level = sorted(Path(p).name for p in list_files(str(root), level=0))
        assert top_level == ["README", "image.JPG"]


def test_walk_to_level_does_not_strip_root():
    gen = walk_to_level("/", level=0)
    root, _, _ = next(gen)
    assert root == "/"

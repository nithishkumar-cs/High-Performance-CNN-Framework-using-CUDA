"""Standard-library tests for exact Darknet checkpoint snapshots."""

import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from monitor_darknet_run import images_seen, snapshot_exact, wait_for_weights  # noqa: E402


class MonitorDarknetRunTest(unittest.TestCase):
    def test_header_and_exact_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "backup.weights"
            destination = root / "snapshot.weights"
            content = struct.pack("<iiiQ", 0, 2, 0, 3200) + b"sample weights"
            source.write_bytes(content)
            self.assertEqual(images_seen(source), 3200)
            wait_for_weights(source, 3200, len(content), time.monotonic() + 4, 0.01, 0)
            snapshot_exact(source, destination, 3200, len(content))
            self.assertEqual(destination.read_bytes(), content)
            with self.assertRaises(FileExistsError):
                snapshot_exact(source, destination, 3200, len(content))

    def test_rejects_wrong_header(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong.weights"
            source.write_bytes(struct.pack("<iiiQ", 1, 0, 0, 3200))
            with self.assertRaises(ValueError):
                images_seen(source)


if __name__ == "__main__":
    unittest.main()

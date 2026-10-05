"""Checks for reproducible VOC XML-to-YOLO conversion."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_voc2007 import parse_annotation, prepare, split_ids, verify_existing  # noqa: E402


XML = """<annotation><size><width>100</width><height>50</height></size>
<object><name>person</name><difficult>0</difficult><bndbox>
<xmin>11</xmin><ymin>6</ymin><xmax>30</xmax><ymax>25</ymax>
</bndbox></object>
<object><name>bird</name><difficult>1</difficult><bndbox>
<xmin>40</xmin><ymin>10</ymin><xmax>50</xmax><ymax>20</ymax>
</bndbox></object></annotation>"""


class ConvertVoc2007Test(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "VOC2007"
        for name in ("ImageSets/Main", "JPEGImages", "Annotations"):
            (source / name).mkdir(parents=True)
        (source / "ImageSets/Main/trainval.txt").write_text(
            "000001\n000002\n000003\n000004\n", encoding="utf-8")
        (source / "ImageSets/Main/test.txt").write_text(
            "000005\n000006\n", encoding="utf-8")
        for value in range(1, 7):
            name = f"{value:06d}"
            (source / "JPEGImages" / f"{name}.jpg").write_bytes(b"jpeg fixture")
            (source / "Annotations" / f"{name}.xml").write_text(XML, encoding="utf-8")
        return source

    def test_coordinates_and_difficult_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "sample.xml"
            xml.write_text(XML, encoding="utf-8")
            self.assertEqual(parse_annotation(xml),
                             (["14 0.20000000 0.30000000 0.20000000 0.40000000"], 1))

    def test_fixed_split_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            split = split_ids(source, 42, 0.25)
            self.assertEqual({name: len(ids) for name, ids in split.items()},
                             {"train": 3, "val": 1, "test": 2})
            self.assertEqual(len(set(sum(split.values(), []))), 6)
            output = root / "yolo"
            summary = prepare(source, output, 42, 0.25)
            self.assertEqual(summary["images"], {"train": 3, "val": 1, "test": 2})
            self.assertEqual(summary["objects"], {"train": 3, "val": 1, "test": 2})
            self.assertEqual(summary["difficult_objects_excluded"], 6)
            self.assertEqual(verify_existing(source, output, 42, 0.25), summary)
            for name, ids in split.items():
                for image_id in ids:
                    self.assertTrue((output / "images" / name / f"{image_id}.jpg").is_file())
                    self.assertEqual((output / "labels" / name / f"{image_id}.txt")
                                     .read_text(encoding="utf-8"),
                                     "14 0.20000000 0.30000000 0.20000000 0.40000000\n")
            with self.assertRaises(ValueError):
                prepare(source, output, 42, 0.25)
            bad = output / "labels" / "test" / "000005.txt"
            bad.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_existing(source, output, 42, 0.25)


if __name__ == "__main__":
    unittest.main()


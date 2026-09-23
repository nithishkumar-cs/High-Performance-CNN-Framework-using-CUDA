import tempfile
import unittest
from pathlib import Path

from tools.prepare_voc2007 import parse_annotation


class VocConversionTests(unittest.TestCase):
    def test_one_based_inclusive_box_and_difficult_filter(self):
        xml = (
            "<annotation><size><width>100</width><height>50</height></size>"
            "<object><name>person</name><difficult>0</difficult>"
            "<bndbox><xmin>1</xmin><ymin>1</ymin>"
            "<xmax>100</xmax><ymax>50</ymax></bndbox></object>"
            "<object><name>cat</name><difficult>1</difficult>"
            "<bndbox><xmin>20</xmin><ymin>10</ymin>"
            "<xmax>30</xmax><ymax>20</ymax></bndbox></object></annotation>"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.xml"
            path.write_text(xml, encoding="utf-8")
            labels, ignored = parse_annotation(path)
        self.assertEqual(labels, ["14 0.50000000 0.50000000 1.00000000 1.00000000"])
        self.assertEqual(ignored, 1)


if __name__ == "__main__":
    unittest.main()

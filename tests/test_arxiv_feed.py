import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_papers import parse_feed  # noqa: E402

FEED = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2312.10997v5</id>
    <published>2023-12-18T00:00:00Z</published>
    <title>Retrieval-Augmented
      Generation Survey</title>
    <summary>A survey.</summary>
    <author><name>Yunfan Gao</name></author>
    <author><name>Another Author</name></author>
  </entry>
</feed>"""


def test_parse_feed():
    (paper,) = parse_feed(FEED)
    assert paper.arxiv_id == "2312.10997"
    assert paper.title == "Retrieval-Augmented Generation Survey"
    assert paper.authors == ["Yunfan Gao", "Another Author"]
    assert paper.year == 2023

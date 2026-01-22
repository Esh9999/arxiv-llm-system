from app import parse_arxiv_atom


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <published>2023-01-15T00:00:00Z</published>
    <title>  Test Title </title>
    <summary> Test abstract. </summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
    <link title="pdf" href="https://arxiv.org/pdf/2301.12345.pdf" type="application/pdf"/>
  </entry>
</feed>
"""

def test_parse_arxiv_atom_basic():
    arts = parse_arxiv_atom(SAMPLE_XML)
    assert len(arts) == 1
    a = arts[0]
    assert a.arxiv_id == "2301.12345"
    assert a.title == "Test Title"
    assert a.authors == ["Alice", "Bob"]
    assert a.abstract == "Test abstract."
    assert "cs.AI" in a.categories
    assert a.published == "2023-01-15"
    assert a.pdf_url.endswith(".pdf")

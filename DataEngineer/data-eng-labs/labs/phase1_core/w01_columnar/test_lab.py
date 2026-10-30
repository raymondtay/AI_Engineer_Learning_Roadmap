import os
import tempfile

import pyarrow.parquet as pq
import pytest

from common.data import docs_table


@pytest.fixture(scope="module")
def layouts(request):
    from conftest import load_lab

    lab = load_lab(str(request.fspath))
    d = tempfile.mkdtemp()
    t = docs_table(4000)
    return lab, t, lab.write_layouts(t, d, row_group_size=500)


def test_all_layouts_agree(layouts):
    lab, t, paths = layouts
    truth = sum(1 for x in t["lang"].to_pylist() if x == "de")
    for layout, p in paths.items():
        assert lab.count_lang(p, layout, "de")[0] == truth, layout
        if layout != "jsonl":
            assert lab.dataset_filter_count(p, "de") == truth


def test_sorted_file_prunes_row_groups(layouts):
    lab, _, paths = layouts
    total = pq.ParquetFile(paths["parquet_sorted"]).metadata.num_row_groups
    kept_sorted = lab.row_groups_matching(paths["parquet_sorted"], "lang", "de")
    kept_unsorted = lab.row_groups_matching(paths["parquet_zstd"], "lang", "de")
    assert len(kept_unsorted) == total          # every random row group contains some 'de'... or could
    assert len(kept_sorted) <= 2                # sorted: 'de' sits in one or two row groups


def test_bytes_scanned_ordering(layouts):
    lab, _, paths = layouts
    js = lab.count_lang(paths["jsonl"], "jsonl", "de")[1]
    unsorted = lab.count_lang(paths["parquet_zstd"], "parquet_zstd", "de")[1]
    srt = lab.count_lang(paths["parquet_sorted"], "parquet_sorted", "de")[1]
    assert js == os.path.getsize(paths["jsonl"])
    assert srt < unsorted < js / 50   # projection alone cuts >50x; sorting cuts more

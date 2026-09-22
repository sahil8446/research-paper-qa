from rag.vectorstore import point_id


def test_point_id_is_stable_and_unique():
    assert point_id("2401.00001:0") == point_id("2401.00001:0")
    assert point_id("2401.00001:0") != point_id("2401.00001:1")

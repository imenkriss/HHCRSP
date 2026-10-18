from pathlib import Path

import pytest

from optimisation.mtwmo.loaders import load_benchmark

ROOT = Path(__file__).resolve().parents[1]


def raw_customer_rows(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    in_customers = False
    rows = []
    for line in lines:
        if line.strip().upper().startswith("CUSTOMER"):
            in_customers = True
            continue
        if not in_customers:
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) >= 7:
            rows.append(values)
    return rows


@pytest.mark.parametrize("relative_path", [
    "data/Solomon/In/c101.txt",
    "data/Gehing&Homberger/200-custmers/C1_2_1.TXT",
])
def test_loader_counts_depot_and_first_last_windows(relative_path):
    path = ROOT / relative_path
    rows = raw_customer_rows(path)
    instance = load_benchmark(path)

    assert len(instance.patients) == len(rows) - 1
    assert instance.depot["id"] == rows[0][0] == 0
    assert (instance.depot["x"], instance.depot["y"]) == (rows[0][1], rows[0][2])
    assert instance.patients[0].id == int(rows[1][0])
    assert instance.patients[0].time_windows == [(rows[1][4], rows[1][5])]
    last = rows[-1]
    assert instance.patients[-1].id == int(last[0])
    assert instance.patients[-1].time_windows == [(last[4], last[5])]

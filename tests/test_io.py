from __future__ import annotations

import threading

from biohub_tracker.io import atomic_write_json


def test_immutable_json_publish_has_exactly_one_concurrent_winner(tmp_path):
    target = tmp_path / "single-use.json"
    barrier = threading.Barrier(2)
    winners = []
    errors = []

    def publish(value):
        barrier.wait()
        try:
            atomic_write_json(target, {"value": value})
            winners.append(value)
        except FileExistsError:
            errors.append(value)

    threads = [threading.Thread(target=publish, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(errors) == 1
    assert target.exists()

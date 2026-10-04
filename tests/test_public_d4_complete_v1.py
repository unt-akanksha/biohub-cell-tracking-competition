"""Tests for the deployable complete-D4 candidate builder.

These assert the artifact contract, not a quality gain. No GPU, Kaggle call,
label, prediction or score is involved.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "public_d4_complete_v1", ROOT / "research" / "public_d4_complete_v1.py"
)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

requires_sources = pytest.mark.skipif(
    not builder.SUPPORT_SOURCE.exists() or not builder.SOURCE_DIR.exists(),
    reason="Pinned public source cache not installed",
)


def test_geometry_proof_is_the_documented_claim():
    report = builder._d4.geometry_check()
    assert report["passed"] and report["model_calls_per_arm"] == 8
    assert not report["quality_gain_established"]
    for row in report["records"]:
        assert row["unique_views"] == {"legacy": 7, "corrected": 8}


def test_unknown_base_rejected():
    with pytest.raises(ValueError, match="Unknown base"):
        builder.load_base("not-a-real-base")


@requires_sources
def test_base_drift_fails_closed(tmp_path, monkeypatch):
    spec = dict(builder.APPROVED_BASES["harmonic"])
    spec["sha256"] = "0" * 64
    monkeypatch.setitem(builder.APPROVED_BASES, "harmonic", spec)
    with pytest.raises(ValueError, match="Public base drift"):
        builder.load_base("harmonic")


@requires_sources
@pytest.mark.parametrize("base", sorted(builder.APPROVED_BASES))
@pytest.mark.parametrize("run_mode", builder.RUN_MODES)
def test_build_verifies_for_every_base_and_mode(base, run_mode):
    # Some bases are qualified for a subset of modes: the sweep and harvest
    # modes patch cells by index and harmonicv3 has a different layout, so the
    # builder refuses those rather than patching the wrong cell.
    spec = builder.APPROVED_BASES[base]
    supported = spec.get("supported_run_modes")
    if supported is not None and run_mode not in supported:
        with pytest.raises(ValueError, match="supports only"):
            builder.build_candidate(base=base, run_mode=run_mode, with_d4=spec.get("supports_d4", True))
        return
    if not spec.get("supports_d4", True):
        # This base is production-only and carries no D4 arm; verify it refuses
        # the correction, then check the build it does support.
        with pytest.raises(ValueError, match="does not support the D4 correction"):
            builder.build_candidate(base=base, run_mode=run_mode, with_d4=True)
        notebook, manifest = builder.build_candidate(
            base=base, run_mode=run_mode, with_d4=False,
            with_density=run_mode in ("densitysweep", "edgeconfsweep"),
            with_edge_confidence=run_mode == "edgeconfsweep",
        )
        checks = builder.verify_built_notebook(notebook, manifest)
        assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)
        assert manifest["d4_correction_applied"] is False
        return
    # densitysweep exists only to compare density configurations, so the
    # mechanism it sweeps has to be installed for the mode to mean anything.
    notebook, manifest = builder.build_candidate(
        base=base, run_mode=run_mode,
        with_density=run_mode in ("densitysweep", "edgeconfsweep"),
        with_edge_confidence=run_mode == "edgeconfsweep",
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)
    assert manifest["total_argument_edits"] == 8
    assert len(manifest["predictor_edits"]) == 6
    assert len(manifest["deepcenter_edits"]) == 2
    assert manifest["proxy_sweep_selection_enabled"] is False
    assert manifest["leaderboard_feedback_used_for_configuration"] is False
    assert manifest["quality_gain_established"] is False


@requires_sources
def test_only_rotation_arguments_differ_in_the_predictor():
    """Byte-exact proof that nothing but the rotation arguments moved.

    Every edit the corrector reports is a single ``1``/``-1`` -> ``2``/``-2``
    argument span. Both replacements preserve byte length, so restoring those
    spans must reproduce the legacy predictor byte for byte. Any other change
    anywhere in the 51 KB dynamically materialized source would survive the
    restore and fail this comparison.
    """
    notebook, manifest = builder.build_candidate(base="harmonic", run_mode="production")
    legacy = builder._d4.materialize_public_predictor(
        builder.load_base("harmonic")[0], builder.SUPPORT_SOURCE.read_bytes()
    )
    corrected = builder._extract_corrected_literal(builder._d4.cell_text(notebook, 4))
    assert builder.sha256(corrected.encode()) == manifest["corrected_predictor_sha256"]
    assert corrected != legacy, "correction did not change anything"

    _, edits = builder._d4.correct_antidiagonal(legacy, predictor=True)
    assert len(edits) == 6

    legacy_bytes = legacy.encode("utf-8")
    corrected_bytes = corrected.encode("utf-8")
    assert len(legacy_bytes) == len(corrected_bytes)

    restored = bytearray(corrected_bytes)
    for edit in edits:
        before = legacy_bytes[edit["start"]:edit["end"]].decode()
        after = corrected_bytes[edit["start"]:edit["end"]].decode()
        if edit["direction"] == "forward":
            assert (before, after) == ("1", "2"), (before, after)
        else:
            assert (before, after) == ("-1", "-2"), (before, after)
        restored[edit["start"]:edit["end"]] = before.encode()

    assert bytes(restored) == legacy_bytes, (
        "the corrected predictor differs from the legacy source outside the "
        "recorded rotation-argument spans"
    )


@requires_sources
def test_deepcenter_cell_differs_only_in_rotation_arguments():
    notebook, _ = builder.build_candidate(base="harmonic", run_mode="production")
    base_nb, _ = builder.load_base("harmonic")
    before = builder._d4.cell_text(base_nb, 5).splitlines()
    after = builder._d4.cell_text(notebook, 5).splitlines()
    assert len(before) == len(after)
    diff = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(diff) == 2
    for a, b in diff:
        normalized = b.replace("rot90(tensor, 2,", "rot90(tensor, 1,").replace(
            "transpose(-1, -2), -2,", "transpose(-1, -2), -1,"
        )
        assert normalized == a


@requires_sources
def test_configuration_constants_are_untouched():
    """No threshold, weight or post-process constant may change."""
    import re

    notebook, _ = builder.build_candidate(base="harmonic", run_mode="production")
    base_nb, _ = builder.load_base("harmonic")
    pattern = r'os\.environ\["(BIOHUB_[A-Z0-9_]+)"\]\s*=\s*[\'"]([^\'"]*)[\'"]'

    def constants(nb, cell):
        return dict(re.findall(pattern, builder._d4.cell_text(nb, cell)))

    before = constants(base_nb, 0)
    after = constants(notebook, 0)
    added = {k: v for k, v in after.items() if k not in before}
    # Only project-authored control keys may be added, never a scientific constant.
    assert set(added) <= {"BIOHUB_VALIDATOR_ENABLE", "BIOHUB_VALIDATOR_N_PER_TYPE",
                          "BIOHUB_D4_RUN_MODE"}
    # An override must CHANGE an existing audited constant, never add a new one,
    # and must be declared in the manifest.
    nb2, man2 = builder.build_candidate(
        base="harmonic", run_mode="production",
        env_overrides={"BIOHUB_DET_THRESHOLD": "0.97"},
    )
    after2 = constants(nb2, 0)
    assert set(after2) - set(before) <= {"BIOHUB_VALIDATOR_ENABLE",
                                         "BIOHUB_VALIDATOR_N_PER_TYPE", "BIOHUB_D4_RUN_MODE"}
    assert after2["BIOHUB_DET_THRESHOLD"] == "0.97"
    assert man2["env_overrides"] == {"BIOHUB_DET_THRESHOLD": "0.97"}
    assert builder.verify_built_notebook(nb2, man2)["no_undeclared_env_writes"]

    for bad in ({"BIOHUB_NOT_AUDITED": "1"},):
        with pytest.raises(ValueError, match="not an audited"):
            builder.build_candidate(env_overrides=bad)
    for bad in ({"BIOHUB_DET_THRESHOLD": "0.1"}, {"BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.99"}):
        with pytest.raises(ValueError, match="outside the audited range"):
            builder.build_candidate(env_overrides=bad)
    for key, value in before.items():
        assert after[key] == value, f"{key} changed from {value} to {after[key]}"


@requires_sources
def test_production_mode_disables_proxy_selection():
    notebook, _ = builder.build_candidate(base="harmonic", run_mode="production")
    cell0 = builder._d4.cell_text(notebook, 0)
    assert '"BIOHUB_VALIDATOR_ENABLE"] = "0"' in cell0


@requires_sources
def test_validation_mode_scores_base_but_selects_nothing():
    notebook, _ = builder.build_candidate(base="harmonic", run_mode="validation")
    assert '"BIOHUB_VALIDATOR_ENABLE"] = "1"' in builder._d4.cell_text(notebook, 0)
    cell10 = builder._d4.cell_text(notebook, 10)
    assert "PP_CANDIDATES: dict[str, dict] = {}" in cell10
    tree = ast.parse(cell10)
    assigned = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PP_CANDIDATES"
    ]
    assert len(assigned) == 1 and ast.literal_eval(assigned[0].value) == {}


@requires_sources
def test_two_device_guard_present_and_fails_closed():
    notebook, _ = builder.build_candidate(base="harmonic", run_mode="production")
    cell0 = builder._d4.cell_text(notebook, 0)
    assert "BIOHUB_D4_DEVICE_GUARD" in cell0
    assert "_d4_devices != 2" in cell0
    assert "raise RuntimeError(" in cell0


@requires_sources
def test_correction_is_not_idempotent_and_rejects_a_corrected_base():
    """Building on an already-corrected predictor must fail, not double-apply."""
    legacy = builder._d4.materialize_public_predictor(
        builder.load_base("harmonic")[0], builder.SUPPORT_SOURCE.read_bytes()
    )
    corrected, _ = builder._d4.correct_antidiagonal(legacy, predictor=True)
    with pytest.raises(ValueError, match="already corrected"):
        builder._d4.correct_antidiagonal(corrected, predictor=True)


@requires_sources
def test_kernel_metadata_is_offline_private_two_t4():
    metadata = builder.kernel_metadata("biohub-d4-harmonic-production-v1", "T")
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert sorted(metadata["dataset_sources"]) == sorted(builder.DATASET_SOURCES)


@requires_sources
def test_built_artifact_on_disk_matches_its_manifest():
    directory = ROOT / "kaggle" / "biohub-d4-harmonic-production-v1"
    if not (directory / "build-manifest.json").exists():
        pytest.skip("artifact not built in this workspace")
    manifest = json.loads((directory / "build-manifest.json").read_text())
    payload = (ROOT / Path(manifest["notebook_path"])).read_bytes()
    assert builder.sha256(payload) == manifest["notebook_sha256"]
    notebook = json.loads(payload)
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)


@requires_sources
def test_artifact_survives_a_windows_push_encoding_round_trip():
    """Regression: the first launch failed because ``kaggle kernels push`` from a
    Windows host re-read the UTF-8 .ipynb with the system code page, turning the
    predictor source's U+00B5 into U+00C2 U+00B5. The on-kernel SHA guard caught
    it. The artifact must now be pure ASCII so no decoder can alter it.
    """
    import hashlib
    import re

    notebook, manifest = builder.build_candidate(base="harmonic", run_mode="production")
    payload = json.dumps(notebook, indent=1, ensure_ascii=True).encode("ascii")
    assert payload.isascii()

    def literal_sha(raw: bytes, decoder: str) -> str:
        parsed = json.loads(raw.decode(decoder))
        cell4 = "".join(parsed["cells"][4]["source"])
        match = re.search(r"^_d4_corrected_source = (.+)$", cell4, flags=re.MULTILINE)
        return hashlib.sha256(ast.literal_eval(match.group(1)).encode("utf-8")).hexdigest()

    want = manifest["corrected_predictor_sha256"]
    for decoder in ("utf-8", "cp1252", "latin-1"):
        assert literal_sha(payload, decoder) == want, decoder

    # Full simulation of the observed corruption path.
    reparsed = json.loads(payload.decode("cp1252"))
    round_tripped = json.dumps(reparsed, ensure_ascii=False).encode("utf-8")
    assert literal_sha(round_tripped, "utf-8") == want


@requires_sources
def test_every_built_cell_is_pure_ascii():
    for run_mode in builder.RUN_MODES:
        notebook, _ = builder.build_candidate(
            base="harmonic", run_mode=run_mode,
            with_density=run_mode in ("densitysweep", "edgeconfsweep"),
            with_edge_confidence=run_mode == "edgeconfsweep",
        )
        for index, cell in enumerate(notebook["cells"]):
            text = "".join(cell["source"])
            assert text.isascii(), f"{run_mode} cell {index} carries non-ASCII"


@requires_sources
def test_recovery_build_is_armed_pinned_and_fail_safe():
    """A recovery build that does not arm the switch is a silent no-op."""
    notebook, manifest = builder.build_candidate(
        base="harmonic", run_mode="production", with_recovery=True
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)

    report = manifest["pruned_track_recovery"]
    assert report["logic_reimplemented"] is False
    assert report["thresholds_retuned"] is False
    assert report["fail_safe_fallback"] is True

    cell0 = builder._d4.cell_text(notebook, 0)
    cell5 = builder._d4.cell_text(notebook, 5)
    assert '"BIOHUB_D4R_ENABLE"] = "1"' in cell0
    assert report["recovery_sha256"] in cell5
    assert "D4R recovery SKIPPED" in cell5, "fail-safe fallback missing"
    assert "_D4R_FROZEN" in cell5, "frozen-constant guard missing"


@requires_sources
def test_recovery_absent_by_default_and_only_touches_two_cells():
    plain, plain_manifest = builder.build_candidate(base="harmonic", run_mode="production")
    assert plain_manifest["pruned_track_recovery"] is None
    assert '"BIOHUB_D4R_ENABLE"] = "1"' not in builder._d4.cell_text(plain, 0)

    withr, _ = builder.build_candidate(
        base="harmonic", run_mode="production", with_recovery=True
    )
    differing = [
        index
        for index, (a, b) in enumerate(zip(plain["cells"], withr["cells"]))
        if "".join(a["source"]) != "".join(b["source"])
    ]
    assert differing == [0, 5], differing


@requires_sources
def test_recovery_embeds_the_scored_implementation_verbatim():
    """The embedded source must be the file the 2026-09-10 run pinned."""
    import hashlib
    import re

    notebook, manifest = builder.build_candidate(
        base="harmonic", run_mode="production", with_recovery=True
    )
    cell5 = builder._d4.cell_text(notebook, 5)
    match = re.search(r"^_D4R_SOURCE = (.+)$", cell5, flags=re.MULTILINE)
    assert match
    embedded = ast.literal_eval(match.group(1))
    assert hashlib.sha256(embedded.encode("utf-8")).hexdigest() == manifest[
        "pruned_track_recovery"
    ]["recovery_sha256"]
    on_disk = (ROOT / "research/public_pruned_track_recovery.py").read_text(encoding="utf-8")
    assert embedded == on_disk


@requires_sources
def test_density_switches_are_sweepable_and_default_to_the_shipped_values():
    """The sweep may only rebind names the notebook agrees are sweepable.

    `pp_apply` raises KeyError for any key absent from PP_SWEEP_KEYS, so an
    unregistered switch would abort the kernel rather than silently scoring the
    base configuration four times and reporting a dead heat.
    """
    notebook, manifest = builder.build_candidate(
        run_mode="densitysweep", with_density=True
    )
    cell5 = builder._d4.cell_text(notebook, 5)
    cell9 = builder._d4.cell_text(notebook, 9)

    for name in builder._density.SWEEPABLE_GLOBALS:
        assert f"{name} = " in cell5
        assert repr(name) in cell9

    # Registration must come after the stock list, never replace part of it.
    for stock in ("MOTION_RELINK_TIGHT_UM", "OUTPUT_EDGE_MAX_UM", "GAP_CLOSE_UM"):
        assert f'"{stock}"' in cell9

    # PP_BASE_CONFIG coerces with type(), so both must already be floats, and
    # the emitted value must track the module constant rather than a literal
    # pinned here -- the low band has already moved once on evidence.
    low = builder._density.DENSITY_BANDS["low"]
    assert isinstance(low, float)
    assert f"DENSITY_LOW_BAND = {low!r}" in cell5
    # Continuous tight is measured-negative and must never ship armed.
    assert "DENSITY_TIGHT_CONTINUOUS = 0.0" in cell5

    swept = set()
    for override in manifest["density_candidates"].values():
        swept |= set(override)
    assert swept <= set(builder._density.SWEEPABLE_GLOBALS)


@requires_sources
def test_continuous_build_actually_ships_the_line():
    """A continuous build that leaves the switch at 0.0 is a silent no-op."""
    banded, _ = builder.build_candidate(with_density=True, density_continuous=False)
    contin, manifest = builder.build_candidate(with_density=True, density_continuous=True)

    assert "DENSITY_TIGHT_CONTINUOUS = 0.0" in builder._d4.cell_text(banded, 5)
    assert "DENSITY_TIGHT_CONTINUOUS = 1.0" in builder._d4.cell_text(contin, 5)
    assert manifest["density_adaptive"]["tight_um_is_continuous"] is True

    checks = builder.verify_built_notebook(contin, manifest)
    assert checks["density_continuous_default_matches_manifest"]

    # And the verifier must reject a manifest/notebook disagreement.
    lying = json.loads(json.dumps(manifest))
    lying["density_adaptive"]["tight_um_is_continuous"] = False
    assert not builder.verify_built_notebook(contin, lying)[
        "density_continuous_default_matches_manifest"
    ]


@requires_sources
def test_density_continuous_requires_the_density_mechanism():
    with pytest.raises(ValueError, match="density_continuous requires"):
        builder.build_candidate(density_continuous=True)
    with pytest.raises(ValueError, match="densitysweep requires"):
        builder.build_candidate(run_mode="densitysweep")


def test_continuous_tight_reproduces_the_published_steps_and_stays_clamped():
    """The line is a fit to the published table, not a new set of constants."""
    density = builder._density
    line = density.CONTINUOUS_TIGHT

    def tight(per_frame):
        raw = line["intercept"] + line["slope"] * per_frame
        return min(line["max"], max(line["min"], raw))

    # Band midpoints the line was fitted through, and the middle band it must
    # reproduce without having been fitted to it.
    assert abs(tight(90.0) - 7.25) < 0.001
    assert abs(tight(470.0) - 5.50) < 0.001
    assert abs(tight(250.0) - 6.50) < 0.02

    # Monotone and clamped to the range the 40-movie set actually covers.
    values = [tight(d) for d in range(1, 1000)]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    assert min(values) >= line["min"] - 1e-9 and max(values) <= line["max"] + 1e-9
    # Observed density range 50.5-557.8: the clamp must not bind inside it.
    assert line["min"] < tight(557.8) and tight(50.5) < line["max"]


@requires_sources
def test_sweep_key_registration_fails_closed_on_an_unexpected_list():
    with pytest.raises(ValueError, match="PP_SWEEP_KEYS anchor"):
        builder._density.register_sweep_keys("PP_SWEEP_KEYS = []\n")


def test_loosen_mode_never_tightens_below_the_published_step():
    """Mode 2's whole claim is that the line may loosen but never tighten."""
    import contextlib
    import io

    density = builder._density
    namespace: dict = {}
    exec(compile(density.preamble(False), "p", "exec"), namespace)
    apply_group = namespace["_apply_density_group"]

    def tight(per_frame, mode):
        namespace["DENSITY_TIGHT_CONTINUOUS"] = float(mode)
        namespace["DENSITY_LOW_BAND"] = 165.0
        nodes = {i: {"t": i % 100} for i in range(int(round(per_frame * 100)))}
        stats: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            apply_group(nodes, stats, "test")
        return stats["density_tight_um_milli"] / 1000.0

    for per_frame in range(5, 1000, 5):
        banded, line, loosen = (tight(per_frame, m) for m in (0, 1, 2))
        assert loosen >= banded - 1e-9, (per_frame, loosen, banded)
        assert loosen == pytest.approx(max(banded, line))

    # The three modes must be genuinely distinct somewhere, or the sweep is
    # comparing a configuration against itself.
    assert tight(297.5, 1) < tight(297.5, 0) == tight(297.5, 2)
    assert tight(190.1, 0) < tight(190.1, 1) == tight(190.1, 2)


@requires_sources
def test_density_candidates_stay_inside_the_declared_mode_set():
    modes = set(builder._density.CONTINUOUS_MODES.values())
    for label, override in builder.DENSITY_CANDIDATES.items():
        if "DENSITY_TIGHT_CONTINUOUS" in override:
            assert override["DENSITY_TIGHT_CONTINUOUS"] in modes, label
        if "DENSITY_LOW_BAND" in override:
            assert override["DENSITY_LOW_BAND"] > 0.0, label
    # The pilot set has to be a real subset of a 20-per-type run, or the
    # held-out reading of `loosen` is not actually held out.
    assert len(builder.PILOT_STEMS) == 12
    assert len(set(builder.PILOT_STEMS)) == 12
    assert sum(s.startswith("44b6") for s in builder.PILOT_STEMS) == 6


@requires_sources
def test_nodecount_sweep_probes_the_untested_node_count_controls():
    """These two constants decide t_pred, which the official metric penalises."""
    notebook, manifest = builder.build_candidate(
        run_mode="nodecountsweep", with_density=True
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)

    cell9 = builder._d4.cell_text(notebook, 9)
    cell10 = builder._d4.cell_text(notebook, 10)
    keys = {k for ov in builder.NODECOUNT_CANDIDATES.values() for k in ov}
    assert keys == {"OUTPUT_MIN_TRACK_LEN", "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"}
    # Both must already be stock-sweepable; this mode registers nothing new.
    for key in keys:
        assert f'"{key}"' in cell9
    assert not (keys & set(builder._density.SWEEPABLE_GLOBALS))

    # Both directions of each knob, or the run cannot distinguish "node count is
    # a lever" from "we happened to push the wrong way".
    minlens = [ov["OUTPUT_MIN_TRACK_LEN"] for ov in builder.NODECOUNT_CANDIDATES.values()
               if "OUTPUT_MIN_TRACK_LEN" in ov]
    rescues = [ov["SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"]
               for ov in builder.NODECOUNT_CANDIDATES.values()
               if "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB" in ov]
    assert min(minlens) < 6 < max(minlens), minlens
    assert min(rescues) < 0.82 < max(rescues), rescues

    # pp_apply coerces with type(PP_BASE_CONFIG[key]); the track length is an int
    # in the base notebook, so a float here would silently truncate.
    assert all(isinstance(v, int) for v in minlens)
    assert all(isinstance(v, float) for v in rescues)
    assert "gap2step40" not in cell10
    assert manifest["nodecount_candidates"] == dict(builder.NODECOUNT_CANDIDATES)


def test_run_modes_is_a_clean_tuple_of_unique_names():
    """A line-based edit once left a dangling continuation here."""
    assert isinstance(builder.RUN_MODES, tuple)
    assert len(builder.RUN_MODES) == len(set(builder.RUN_MODES))
    assert all(isinstance(m, str) and m and m.islower() for m in builder.RUN_MODES)
    assert {"production", "densitysweep", "nodecountsweep"} <= set(builder.RUN_MODES)


@requires_sources
def test_edge_confidence_floor_is_inert_by_default_and_enforced_when_raised():
    """The floor targets false positives, the dominant error mode.

    It must be a no-op at 0.0 so a production build behaves exactly like the
    shipped 0.949 configuration, and it must be enforced where the assignment
    actually decides, not merely defined.
    """
    plain, plain_manifest = builder.build_candidate(with_density=True)
    assert "MOTION_RELINK_MIN_LEARNED_PROB" not in builder._d4.cell_text(plain, 5)
    assert builder.verify_built_notebook(plain, plain_manifest)[
        "floor_absent_when_not_installed"
    ]

    notebook, manifest = builder.build_candidate(
        run_mode="edgeconfsweep", with_density=True, with_edge_confidence=True
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)

    cell5 = builder._d4.cell_text(notebook, 5)
    assert "MOTION_RELINK_MIN_LEARNED_PROB = 0.0" in cell5
    assert manifest["edge_confidence"]["is_no_op_by_default"] is True

    # Enforced inside assign_pass, after prob, before the cost is written.
    prob_at = cell5.index("prob = learned_prob(source_id, target_id)")
    guard_at = cell5.index("if prob < MOTION_RELINK_MIN_LEARNED_PROB:")
    cost_at = cell5.index("cost[i, j] = motion + 0.05 * raw")
    assert prob_at < guard_at < cost_at

    # The floor must be reachable through pp_apply, which rejects unregistered
    # keys -- otherwise the kernel aborts instead of scoring the candidates.
    cell9 = builder._d4.cell_text(notebook, 9)
    for name in builder._edgeconf.SWEEPABLE_GLOBALS:
        assert repr(name) in cell9
    swept = {k for ov in builder.EDGECONF_CANDIDATES.values() for k in ov}
    assert swept == set(builder._edgeconf.SWEEPABLE_GLOBALS)

    # Both regimes must be probed: at or below the 0.48 inference threshold
    # (model-proposed pairs only) and above it (confident among those).
    vals = [ov["MOTION_RELINK_MIN_LEARNED_PROB"]
            for ov in builder.EDGECONF_CANDIDATES.values()]
    assert any(0.0 < v <= 0.48 for v in vals), vals
    assert any(v > 0.48 for v in vals), vals
    assert all(isinstance(v, float) and 0.0 < v < 1.0 for v in vals)


@requires_sources
def test_edgeconfsweep_requires_the_mechanism():
    with pytest.raises(ValueError, match="edgeconfsweep requires"):
        builder.build_candidate(run_mode="edgeconfsweep", with_density=True)


@requires_sources
def test_production_may_arm_the_floor_only_inside_the_plateau():
    """Any floor in (0, 0.48] is the same switch; above it is a tuned number.

    The predictor emits edges only above its 0.48 inference threshold and pairs
    it never proposed score exactly 0.0, so learned probability lies in
    {0} U (0.48, 1]. Values above 0.48 cut genuine proposals and measured worse.
    """
    notebook, manifest = builder.build_candidate(
        with_density=True, with_edge_confidence=True, edge_confidence_floor=0.10
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)
    assert checks["production_floor_in_plateau"]
    assert "MOTION_RELINK_MIN_LEARNED_PROB = 0.1" in builder._d4.cell_text(notebook, 5)
    assert manifest["edge_confidence"]["min_learned_prob_default"] == 0.10
    assert manifest["edge_confidence"]["is_no_op_by_default"] is False

    for bad in (0.5, 0.8, 1.0, -0.1):
        with pytest.raises(ValueError, match=r"\(0, 0.48\]"):
            builder.build_candidate(
                with_density=True, with_edge_confidence=True, edge_confidence_floor=bad
            )

    with pytest.raises(ValueError, match="edge_confidence_floor requires"):
        builder.build_candidate(with_density=True, edge_confidence_floor=0.10)


@requires_sources
def test_guarded_override_moves_the_publishers_expectation_in_lockstep():
    """Cell 1 carries the publisher's config-drift guard.

    Overriding BIOHUB_DET_THRESHOLD without moving its expectation raised
    RuntimeError on Kaggle and killed the run in 14 seconds. A declared change
    must move both; everything we did not touch must stay exactly as published.
    """
    notebook, manifest = builder.build_candidate(
        with_density=True, env_overrides={"BIOHUB_DET_THRESHOLD": "0.995"}
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)

    cell1 = builder._d4.cell_text(notebook, 1)
    assert '"BIOHUB_DET_THRESHOLD": 0.995,' in cell1
    assert '"BIOHUB_DET_THRESHOLD": 0.965,' not in cell1
    # Every other guarded constant must be untouched.
    for untouched in ('"BIOHUB_GAP_CLOSE_UM": 5.0,',
                      '"BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,',
                      '"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,'):
        assert untouched in cell1

    # A build with no override must leave the guard byte-identical.
    plain, _ = builder.build_candidate(with_density=True)
    base_nb, _ = builder.load_base("harmonic")
    assert builder._d4.cell_text(plain, 1) == builder._d4.cell_text(base_nb, 1)

    # An ungarded key must not touch cell 1 at all.
    edge, _ = builder.build_candidate(
        with_density=True, env_overrides={"BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.40"}
    )
    assert builder._d4.cell_text(edge, 1) == builder._d4.cell_text(base_nb, 1)


@requires_sources
def test_pool_kernel_patch_changes_exactly_one_constant():
    """The detection suppression radius is not reachable from the environment.

    It lives in the support-pack predictor's dataclass default with no CLI flag,
    so it is changed in the source itself. That is only safe if the edit is
    provably confined to one literal and hash-guarded on both sides.
    """
    pk = builder._poolkernel
    legacy = builder._d4.materialize_public_predictor(
        builder.load_base("harmonic")[0], builder.SUPPORT_SOURCE.read_bytes()
    )
    patched = pk.patch_source(legacy, 6.0)

    before, after = legacy.splitlines(), patched.splitlines()
    assert len(before) == len(after)
    diff = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(diff) == 1
    assert "3.0" in diff[0][0] and "6.0" in diff[0][1]
    # The trailing comment documents the units and must survive.
    assert diff[0][0].split("#", 1)[1] == diff[0][1].split("#", 1)[1]
    # Restoring the one literal must reproduce the shipped source exactly.
    assert patched.replace(
        "pool_kernel_um: float = 6.0", "pool_kernel_um: float = 3.0", 1
    ) == legacy

    for bad in (3.0, 2.0, 9.5, 12.0):
        with pytest.raises(ValueError, match="audited range"):
            pk.patch_source(legacy, bad)


@requires_sources
def test_pool_kernel_build_is_guarded_and_refuses_to_combine_with_d4():
    notebook, manifest = builder.build_candidate(
        with_density=True, with_d4=False, pool_kernel_um=6.0
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)

    cell4 = builder._d4.cell_text(notebook, 4)
    assert "PROJECT_POOL_KERNEL" in cell4
    assert manifest["pool_kernel"]["legacy_predictor_sha256"] in cell4
    assert manifest["pool_kernel"]["patched_predictor_sha256"] in cell4
    assert "PROJECT_D4_CORRECTION" not in cell4

    # Both rewrite the same predictor file; the second would invalidate the
    # first's hash guard, so the combination must fail at build time.
    with pytest.raises(ValueError, match="cannot be combined with the D4"):
        builder.build_candidate(with_density=True, with_d4=True, pool_kernel_um=6.0)

    plain, plain_manifest = builder.build_candidate(with_density=True, with_d4=False)
    assert plain_manifest["pool_kernel"] is None
    assert builder.verify_built_notebook(plain, plain_manifest)["pool_kernel_absent"]


@requires_sources
def test_bidirectional_weight_moves_both_of_its_pins():
    """This constant is pinned in TWO places, not one.

    Cell 1 carries the general config-drift guard; cell 4 carries a second
    assertion immediately before the source splice that consumes the value.
    Moving only the first produced a ValueError on Kaggle and cost a run. The
    spliced code reads the environment variable at runtime rather than
    hardcoding it, so both pins are value pins and both must move together.
    """
    notebook, manifest = builder.build_candidate(
        with_density=True, env_overrides={"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.30"}
    )
    checks = builder.verify_built_notebook(notebook, manifest)
    assert all(checks.values()), sorted(n for n, ok in checks.items() if not ok)
    assert checks["bidirectional_pin_synced"] and checks["bidirectional_old_pin_gone"]

    cell1 = builder._d4.cell_text(notebook, 1)
    cell4 = builder._d4.cell_text(notebook, 4)
    assert '"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.3,' in cell1
    assert '"expected_bidirectional_weight": 0.3,' in cell4
    assert "_bidirectional_weight_guard, 0.3," in cell4
    assert "0.15" not in cell4.split("_bidirectional_weight_guard")[1][:400]

    # A build without the override must leave both cells byte-identical.
    # with_d4 must be off here: the D4 correction legitimately writes into
    # cell 4, so leaving it on would compare against a different change.
    base_nb, _ = builder.load_base("harmonic")
    plain, _ = builder.build_candidate(with_density=True, with_d4=False)
    assert builder._d4.cell_text(plain, 1) == builder._d4.cell_text(base_nb, 1)
    assert builder._d4.cell_text(plain, 4) == builder._d4.cell_text(base_nb, 4)

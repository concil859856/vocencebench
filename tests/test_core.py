import io

import numpy as np
import soundfile as sf

import vocencebench as vb
from vocencebench import prompts, traits
from vocencebench.judge.base import Judge
from vocencebench.metrics import bootstrap_ci, win_rate
from vocencebench.probes import default_probes
from vocencebench.schema import Verdict, WINNER_A, WINNER_B, TIE


def _wav(freq=180.0, sec=3.0, amp=0.1, sr=24000):
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    y = amp * np.sin(2 * np.pi * freq * t).astype("float32")
    b = io.BytesIO()
    sf.write(b, y, sr, format="WAV")
    return b.getvalue()


# ------------------------------------------------------------------- traits & schema
def test_trait_registry():
    assert traits.get("gender").kind == "objective"
    assert traits.get("tone").needs_judge()
    assert traits.get("pace").ordinal
    # unknown trait -> holistic fallback
    assert traits.get("whatever").kind == "holistic"


def test_dataset_roundtrip(tmp_path):
    samples = [vb.Sample(id="1", text="hello", traits={"gender": "male"}),
               vb.Sample(id="2", text="world", instruction="a fast voice", traits={"pace": "fast"})]
    p = tmp_path / "d.jsonl"
    vb.save_dataset(samples, p)
    back = vb.load_dataset(p)
    assert [s.id for s in back] == ["1", "2"]
    assert back[1].traits == {"pace": "fast"}


# ------------------------------------------------------------------------ prompts
def test_prompt_parse_and_winner_normalisation():
    for raw, expect in [("a", "a"), ("1", "a"), ("System 2", "b"), ("tie", "tie"), ("?", "tie")]:
        v = prompts.parse_verdict({"reasoning_a": "x", "reasoning_b": "y",
                                   "score_a": 2, "score_b": 3, "winner": raw})
        assert v["winner"] == expect


def test_adherence_prompt_mentions_trait():
    p = prompts.adherence("hi", "a warm voice", "tone", "warm")
    assert "tone" in p.intro and "warm" in p.intro


# -------------------------------------------------------------------------- judge
class _PositionBias:
    """Always picks the first clip it hears."""
    def compare(self, parts, a, b, *, temperature):
        return {"winner": "a", "score_a": 3, "score_b": 1, "comparison": "", "confidence": 1.0}


class _Energy:
    """Order-independent: prefers the higher-energy clip."""
    def compare(self, parts, a, b, *, temperature):
        ea = np.abs(sf.read(io.BytesIO(a))[0]).mean()
        eb = np.abs(sf.read(io.BytesIO(b))[0]).mean()
        w = "a" if ea > eb else "b"
        return {"winner": w, "score_a": 3 if w == "a" else 1,
                "score_b": 1 if w == "a" else 3, "comparison": "", "confidence": 0.8}


def test_order_swap_catches_position_bias():
    j = Judge(_PositionBias(), swap=True)
    v = j.naturalness("hi", _wav(), _wav())
    assert v.winner == TIE and v.consistent is False


def test_order_swap_keeps_consistent_content_winner():
    j = Judge(_Energy(), swap=True)
    v = j.naturalness("hi", _wav(amp=0.2), _wav(amp=0.02))  # clip a louder
    assert v.winner == WINNER_A and v.consistent is True


# ------------------------------------------------------------------------- probes
def test_probes_score_requested_traits():
    s = vb.Sample(id="0", text="one two three four five six seven eight",
                  traits={"loudness": "normal", "pace": "moderate"})
    out = {p.trait: p for pr in default_probes() if (p := pr.score(s, _wav(sec=3.0, amp=0.1)))}
    assert out["loudness"].measured is not None
    assert out["pace"].detail["words_per_sec"] > 0


def test_probe_returns_none_when_trait_absent():
    from vocencebench.probes.acoustic import PitchProbe
    s = vb.Sample(id="0", text="hello", traits={"gender": "male"})
    assert PitchProbe().score(s, _wav()) is None


# ------------------------------------------------------------------------ metrics
def test_win_rate_and_consistency_filter():
    vs = [Verdict("d", WINNER_A, consistent=True),
          Verdict("d", WINNER_B, consistent=True),
          Verdict("d", TIE, consistent=True),
          Verdict("d", WINNER_A, consistent=False)]  # dropped
    assert win_rate(vs) == 0.5  # 1 + 0 + 0.5 over 3


def test_bootstrap_ci_bounds():
    lo, hi = bootstrap_ci([1.0] * 10)
    assert lo == 1.0 and hi == 1.0


# --------------------------------------------------------------------- end to end
def test_classifier_probe_label_mapping():
    from vocencebench.probes.classifier import HFClassifierProbe
    probe = HFClassifierProbe("gender", "fake/model", {"female": "female", "male": "male"})
    # Stub out the pipeline: top label maps to 'male'.
    probe._pipe = lambda audio, top_k=5: [{"label": "MALE", "score": 0.97},
                                          {"label": "female", "score": 0.03}]
    s = vb.Sample(id="0", text="hi", traits={"gender": "male"})
    r = probe.score(s, _wav())
    assert r.measured == "male" and r.matched is True and r.score == 1.0
    # Not-requested trait returns None.
    assert probe.score(vb.Sample(id="1", text="hi", traits={"pace": "fast"}), _wav()) is None


def test_classifier_probe_ordinal_partial_credit():
    from vocencebench.probes.classifier import HFClassifierProbe
    probe = HFClassifierProbe("pitch", "fake", {"low": "low", "mid": "medium", "high": "high"},
                              order=("low", "medium", "high"))
    probe._pipe = lambda audio, top_k=5: [{"label": "mid", "score": 0.8}]
    r = probe.score(vb.Sample(id="0", text="hi", traits={"pitch": "high"}), _wav())
    assert r.measured == "medium" and r.score == 0.5  # adjacent bucket


def test_calibration_metrics():
    from vocencebench import calibration as cal
    assert cal.cohen_kappa(["a", "b", "tie"], ["a", "b", "tie"]) == 1.0
    assert cal.cohen_kappa(["a", "a", "a", "a"], ["a", "b", "a", "b"]) == 0.0
    assert cal.krippendorff_alpha([["a", "a"], ["b", "b"]]) == 1.0
    assert cal.spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_calibration_report_and_weights():
    from vocencebench import calibration as cal
    gold = [cal.GoldItem("0", "naturalness", ["a", "a", "b"], "a"),
            cal.GoldItem("1", "naturalness", ["b", "b", "b"], "b"),
            cal.GoldItem("2", "tone", ["a", "b", "tie"], "tie")]
    rep = cal.agreement_report(gold)
    assert rep["naturalness"]["judge_accuracy"] == 1.0
    w = cal.suggest_weights(rep)
    assert abs(sum(w.values()) - 1.0) < 1e-6 or w == {}


def test_build_splits_disjoint():
    pub, hold = vb.build_splits(20, holdout_every=5)
    ids_pub = {s.id for s in pub}
    ids_hold = {s.id for s in hold}
    assert len(pub) == 16 and len(hold) == 4
    assert ids_pub.isdisjoint(ids_hold)


def test_gold_export_import_roundtrip(tmp_path):
    from vocencebench import calibration as cal
    items = [{"sample_id": "0", "dimension": "naturalness", "category": "general", "difficulty": 0}]
    p = tmp_path / "gold.jsonl"
    cal.export_for_labeling(items, p)
    # simulate a rater filling winner
    lines = p.read_text().splitlines()
    import json
    row = json.loads(lines[0]); row["winner"] = "a"
    p.write_text(json.dumps(row) + "\n")
    gold = cal.load_gold(p)
    assert gold[0].human_labels == ["a"]


def test_evaluate_end_to_end():
    model = lambda t, i: _wav(190, 3.0, 0.12)
    ref = lambda t, i: _wav(170, 3.6, 0.03)
    data = [vb.Sample(id=str(i), text="the old lighthouse stood watch",
                      instruction="warm british man",
                      traits={"pace": "moderate", "loudness": "normal", "tone": "warm"})
            for i in range(3)]
    rep = vb.evaluate(data, model, ref, Judge(_Energy(), swap=True), probes=default_probes())
    assert rep.n == 3
    assert rep.metrics["naturalness_win_rate"] == 1.0  # model louder
    assert "tone" in rep.metrics["trait_csr"]
    assert rep.metrics["errors"] == 0

"""Scope checks protect the new experiment and its fail-closed application seam."""
import json

import numpy as np
import pytest

from intentdesk.scope import ScopeDetector, ScopeRouter, assessment, scope_features
from intentdesk.scope_experiment import split_new_oos, select_thresholds, scope_metrics, group_order


def unit_rows(n):
    rng = np.random.default_rng(10)
    x = rng.normal(size=(n, 768)).astype(np.float32)
    return x/np.linalg.norm(x, axis=1, keepdims=True)


def test_novelty_distance_uses_real_reference_and_entropy():
    ref = unit_rows(150)
    p = np.zeros((1, 150), dtype=np.float32); p[0, 0] = 1
    values = scope_features(ref[:1], ref, ref, p)
    assert values.shape == (1, 6)
    assert values[0, 0] == pytest.approx(1., abs=1e-5)
    assert values[0, 1] < values[0, 0]
    assert values[0, 3] == 1 and values[0, 4] == 1 and values[0, 5] == 0
    uniform = scope_features(ref[:1], ref, ref, np.full((1, 150), 1/150))
    assert uniform[0, 5] == pytest.approx(1.)


def test_invalid_normalization_or_probability_contract_rejected():
    ref = unit_rows(150)
    p = np.full((1, 150), 1/150)
    with pytest.raises(ValueError): scope_features(ref[:1]*2, ref, ref, p)
    with pytest.raises(ValueError): scope_features(ref[:1], ref, ref, p*2)
    with pytest.raises(ValueError): scope_features(ref[:1], ref, ref, p*np.nan)


def test_fresh_split_excludes_all_old_fingerprints_and_keeps_groups_apart():
    from intentdesk.data import fingerprint
    pairs = [[f"fresh phrase number {i}", "oos"] for i in range(151)]
    pairs += [[" OLD phrase! ", "oos"], pairs[0]]
    selection, check, excluded = split_new_oos(pairs, {fingerprint("old phrase")})
    assert len(selection) == 51 and len(check) == 100 and len(excluded) == 2
    assert not {r['duplicate_group'] for r in selection} & {r['duplicate_group'] for r in check}
    assert split_new_oos(pairs, {fingerprint("old phrase")}) == (selection, check, excluded)


def test_selection_falls_back_without_hiding_failure():
    thresholds, trials = select_thresholds(np.full(20, .9), np.full(20, .9))
    assert not thresholds["selection_passed"] and thresholds["upper"] == 1.1
    assert len(trials) == 21
    assert assessment(.89, thresholds['lower'], thresholds['upper']) == 'likely_out_of_scope'
    assert assessment(.9, thresholds['lower'], thresholds['upper']) == 'uncertain'


def test_separated_selection_and_metric_counts():
    t, _ = select_thresholds(np.linspace(.8, .99, 100), np.linspace(.01, .2, 51))
    assert t['selection_passed']
    assert assessment(.01, t['lower'], t['upper']) == 'likely_out_of_scope'
    assert assessment(.99, t['lower'], t['upper']) == 'supported_candidate'
    m = scope_metrics(np.array([.99, .01]), np.array([.01, .99]), t)
    assert m['known_supported_candidates'] == 1 and m['fresh_oos_supported_candidates'] == 1
    assert m['fresh_oos_flagged_likely_oos'] == 1


def test_artifact_path_traversal_rejected(tmp_path):
    folder = tmp_path/'artifacts'; folder.mkdir()
    (folder/'scope_latest.json').write_text(json.dumps({'run_id':'../outside','metadata_sha256':'bad'}))
    with pytest.raises(ValueError, match='artifact path'): ScopeDetector(tmp_path, 'parent')


def test_scope_router_never_routes_or_exposes_wrong_candidates(monkeypatch):
    import intentdesk.scope as module
    router = object.__new__(ScopeRouter)
    router.metadata = {'run_id':'parent','gate':{'passed':True},'temperatures':{'B2_MPNet_linear':{'temperature':1}}}
    router.tokenizer = lambda text, **kwargs: {'input_ids':[1,2]}
    router.encoder = None
    router.prototypes = unit_rows(150)
    class Models:
        classes = ['x']*150
        def logits(self, name, x): return np.zeros((1, 150))
    class Detector:
        def predict(self, *args): return {'status':'likely_out_of_scope'}
    router.models = Models(); router.scope = Detector()
    monkeypatch.setattr(module, 'encode_batch', lambda *args: unit_rows(1))
    r = router.route('a fictional unsupported request')
    assert r['route_recommendation'] == 'review_required' and not r['action_executed']
    assert r['candidates'] == [] and len(r['nearest_known_candidates']) == 3
    assert 'Likely outside' in r['reason']


def test_invalid_optional_model_keeps_review_fallback(tmp_path, monkeypatch):
    import intentdesk.scope as module
    def init(router, root):
        router.metadata = {'run_id':'parent'}
    monkeypatch.setattr(module.Router, '__init__', init)
    monkeypatch.setattr(module.Router, 'route', lambda *a: {
        'route_recommendation':'route_to_intent','action_executed':True,'candidates':[]})
    directory = tmp_path/'artifacts'; directory.mkdir()
    (directory/'scope_latest.json').write_text(json.dumps({'run_id':'../outside','metadata_sha256':'bad'}))
    router = ScopeRouter(tmp_path)
    result = router.route('fictional fixture')
    assert result['scope_assessment']['status'] == 'unavailable'
    assert result['route_recommendation'] == 'review_required' and not result['action_executed']


def test_changed_binding_is_refused_before_check_exposure(tmp_path, monkeypatch):
    import intentdesk.scope_experiment as module
    directory = tmp_path/'artifacts'/module.RUN; directory.mkdir(parents=True)
    (directory/'metadata.json').write_text(json.dumps({'status':'trained_frozen'}))
    marker = tmp_path/'artifacts/scope_v1_experiment.json'
    marker.write_text(json.dumps({'binding_sha256':'old','check_exposed':False}))
    monkeypatch.setattr(module, 'verified_inputs', lambda root: {})
    monkeypatch.setattr(module, 'frozen_binding', lambda *args: 'changed')
    with pytest.raises(ValueError, match='binding changed'): module.evaluate_scope(tmp_path)
    assert json.loads(marker.read_text())['check_exposed'] is False


def test_completed_scope_report_checksum_checked_before_reuse(tmp_path, monkeypatch):
    import intentdesk.scope_experiment as module
    directory = tmp_path/'artifacts'/module.RUN; directory.mkdir(parents=True)
    (directory/'metadata.json').write_text(json.dumps({'status':'completed'}))
    (tmp_path/'artifacts/scope_v1_experiment.json').write_text(json.dumps({
        'binding_sha256':'fixed','status':'completed','report_sha256':'wrong'}))
    reports = tmp_path/'reports'; reports.mkdir()
    (reports/'scope_evaluation.json').write_text('{}')
    monkeypatch.setattr(module, 'verified_inputs', lambda root: {})
    monkeypatch.setattr(module, 'frozen_binding', lambda *args: 'fixed')
    with pytest.raises(ValueError, match='report checksum mismatch'): module.evaluate_scope(tmp_path)

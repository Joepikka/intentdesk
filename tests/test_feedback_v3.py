import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from intentdesk import feedback_v3 as v3
from intentdesk.common import write_json, sha256_file
from intentdesk.feedback_data import DIRECTORY, ingest, product_split, text_key


def test_markup_cleanup_decodes_entities_and_removes_formatting_not_sentiment():
    assert v3.clean_markup('Good<br />but &#34;BROKEN&#34; [[VIDEOID:abcdef]] &amp; expensive') == 'good but "broken" & expensive'
    assert v3.clean_markup('Café &lt;b&gt;great&lt;/b&gt;') == 'café great'
    assert v3.clean_markup('I rate it 2/5') == 'i rate it 2/5'  # rating cleanup remains in the existing data boundary


def test_idf_document_counts_are_not_token_counts_and_output_is_sparse_float32(monkeypatch):
    monkeypatch.setitem(v3.CONFIG, 'features_per_view', 128)
    x = v3.counts(['good good good', 'bad bad', 'good bad'])
    df = v3.document_frequency(x)
    assert np.array_equal(df, np.asarray((x>0).sum(0)).ravel())
    assert df.max() <= 3
    idf = (np.log(4/(1+df))+1).astype(np.float32)
    transformed = v3.idf_features(x, idf)
    assert transformed.dtype == np.float32 and transformed.shape == (3, 256)
    assert np.allclose(np.asarray(transformed.multiply(transformed).sum(1)).ravel(), 1)
    assert x.data.max() >= 3  # transforming did not mutate the count matrix


def test_selection_requires_class_balanced_gain_not_just_majority_accuracy():
    results = {v3.CONTROL: {'macro_f1': .55, 'accuracy': .72},
               'idf_moderate': {'macro_f1': .54, 'accuracy': .91},
               'idf_balanced': {'macro_f1': .554, 'accuracy': .74}}
    assert v3.selection(results) == v3.CONTROL
    results['idf_balanced']['macro_f1'] = .56
    assert v3.selection(results) == 'idf_balanced'


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setitem(v3.CONFIG, 'features_per_view', 128)
    monkeypatch.setitem(v3.CONFIG, 'batch_size', 8)
    monkeypatch.setitem(v3.CONFIG, 'fresh_check_rows', 10)
    monkeypatch.setitem(v3.CONFIG, 'maximum_epochs', 2)
    products = {}
    for i in range(1000):
        p = f'v3-fixture-{i}'
        products.setdefault(product_split(p), []).append(p)
    adjectives = {1:'awful broken terrible', 2:'poor weak disappointing',
                  3:'ordinary adequate average', 4:'good useful reliable', 5:'wonderful excellent superb'}
    fit = [{'text':f'{adjectives[y]} fit product example {i}', 'rating':y, 'product':products['fit'][0]}
           for y in range(1,6) for i in range(8)]
    dev = [{'text':f'{adjectives[y]} unseen development example {i}', 'rating':y, 'product':products['development'][0]}
           for y in range(1,6) for i in range(3)]
    old = [{'text':f'{adjectives[y]} original consumed example {y}', 'rating':y, 'product':products['test'][0]}
           for y in range(1,6)]
    second = [{'text':f'{adjectives[y]} second consumed example {y}', 'rating':y, 'product':products['test'][1]}
              for y in range(1,6)]
    # More reviews of already-tested products must also be excluded.
    extra = [{'text':f'{adjectives[y]} unselected review on consumed products', 'rating':y, 'product':products['test'][i]}
             for i in (0,1) for y in range(1,6)]
    fresh = [{'text':f'{adjectives[y]} fresh product example {i}', 'rating':y, 'product':products['test'][2]}
             for y in range(1,6) for i in range(4)]
    for row in fit+dev+old+second+extra+fresh:
        row['id'] = text_key(row['text'])
    data = tmp_path/DIRECTORY
    data.mkdir(parents=True)
    source = data/'source.jsonl'
    source.write_text(''.join(json.dumps({'title':'','text':r['text'],'rating':r['rating'],'parent_asin':r['product']})+'\n'
                             for r in fit+dev+old+second+extra+fresh))
    ingest([source], data/'audit.sqlite3')
    manifest = {'audit_sha256':sha256_file(data/'audit.sqlite3'),
                'available_by_partition_and_rating':{'fit':{str(y):8 for y in range(1,6)}}}
    write_json(data/'manifest.json', manifest)
    old_dir = tmp_path/'artifacts/original'
    old_dir.mkdir(parents=True)
    write_json(old_dir/'final_experiment.json', {'status':'completed'})
    v2_dir = tmp_path/'artifacts/v2'
    v2_dir.mkdir()
    (v2_dir/'fresh_check.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in second))
    md = {'selected_classifier':'word_char_balanced',
          'protocol':{'fresh_check':{'sha256':sha256_file(v2_dir/'fresh_check.jsonl')}}}
    write_json(v2_dir/'metadata.json', md)
    write_json(v2_dir/'final_experiment.json', {'status':'completed'})
    monkeypatch.setattr(v3, 'directory_for', lambda root:old_dir)
    monkeypatch.setattr(v3, 'checked_metadata', lambda root:{'data':manifest})
    monkeypatch.setattr(v3.previous, 'current_directory', lambda root:v2_dir)
    monkeypatch.setattr(v3.previous, 'checked', lambda root:(v2_dir,md))
    monkeypatch.setattr(v3, 'load_partition', lambda root,name:{'fit':fit,'development':dev,'test':old}[name])
    monkeypatch.setattr(v3, 'binding', lambda root:{'fixture':True,'config':v3.CONFIG.copy()})
    model = SGDClassifier(loss='log_loss', random_state=12).fit(v3.previous.features([r['text'] for r in fit]), [r['rating'] for r in fit])
    joblib.dump({'word_char_balanced':model}, v2_dir/'models.joblib')
    return tmp_path,old_dir,v2_dir,fit,fresh


def test_primary_key_stream_has_stable_order_and_exact_resume_without_temp_sort(toy):
    root,_,_,fit,_ = toy
    rows = [r for batch in v3.fit_rows(root) for r in batch]
    assert [r['id'] for r in rows] == sorted(r['id'] for r in fit)
    after = rows[10]['id']
    assert [r['id'] for batch in v3.fit_rows(root,after) for r in batch] == [r['id'] for r in rows[11:]]
    with sqlite3.connect(root/DIRECTORY/'audit.sqlite3') as db:
        plan = db.execute("EXPLAIN QUERY PLAN SELECT hash,text FROM reviews INDEXED BY sqlite_autoindex_reviews_1 WHERE hash>? AND split='fit' ORDER BY hash", ('',)).fetchall()
    assert not any('TEMP B-TREE' in str(r) for r in plan)


def test_new_check_excludes_all_products_in_both_consumed_checks(toy):
    root,_,_,_,fresh = toy
    target = root/'reserved'
    target.mkdir()
    check = v3.reserve_check(root,target)
    rows = [json.loads(line) for line in (target/'fresh_check.jsonl').read_text().splitlines()]
    assert check['blocked_previous_products'] == 2 and check['previous_product_overlap'] == 0
    assert {r['product'] for r in rows} == {fresh[0]['product']}


def test_real_toy_training_reload_one_final_check_and_preservation(toy):
    root,old,v2,_,_ = toy
    before = {str(p):sha256_file(p) for directory in (old,v2) for p in directory.iterdir()}
    metadata = v3.train(root)
    assert metadata['idf_fit_rows'] == 40 and metadata['completed_epochs'] == 2
    assert metadata['test_exposed'] is False
    assert set(metadata['development']) == {v3.CONTROL,'idf_balanced','idf_moderate'}
    predicted = v3.FeedbackV3(root).predict(['terrible broken product', 'excellent useful product'])
    assert len(predicted) == 2 and all(p in range(1,6) for p in predicted)
    report = v3.evaluate(root)
    assert report['selected_classifier'] == metadata['selected_classifier']
    assert report['selection_unchanged_after_test'] and v3.evaluate(root) == report
    assert before == {str(p):sha256_file(p) for directory in (old,v2) for p in directory.iterdir()}
    path = v3.current_directory(root)/'fresh_check.jsonl'
    path.write_text('tampered')
    with pytest.raises(ValueError,match='reserved check changed'):
        v3.evaluate(root)


def test_source_change_refuses_resume_instead_of_silently_reusing_weights(toy,monkeypatch):
    root,*_ = toy
    v3.prepare(root)
    monkeypatch.setattr(v3, 'binding', lambda root:{'fixture':'changed'})
    with pytest.raises(ValueError,match='source/data/config changed'):
        v3.prepare(root)

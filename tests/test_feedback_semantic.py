import copy
import json
from pathlib import Path
import types
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from intentdesk import feedback_semantic as sem
from intentdesk import feedback_semantic_data as data
from intentdesk import feedback_semantic_encoder as enc
from intentdesk import feedback_semantic_models as models
from intentdesk.common import read_json,write_json,sha256_file
from intentdesk.feedback_data import DIRECTORY,ingest,product_split,text_key


def test_ordered_head_keeps_five_classes_with_monotone_thresholds_and_gradients():
    head=models.OrderedHead(8)
    with torch.no_grad(): head.gaps.copy_(torch.tensor([5.,-8.,1.]))
    x=torch.randn(10,8)
    y=torch.tensor([1,2,3,4,5]*2)
    p=head.probabilities(x)
    assert p.shape==(10,5) and (p>=0).all() and torch.allclose(p.sum(1),torch.ones(10))
    assert (head.thresholds().diff()>0).all()
    assert (head(x).sigmoid().diff(dim=1)<=0).all()
    loss=models.loss_rows(head,x,y).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert torch.allclose(models.rating_weights(np.array([1,2,3,4,5]*2)),torch.ones(5))


def test_ordered_targets_learn_low_and_high_ratings_without_merging_classes():
    torch.manual_seed(11)
    head=models.OrderedHead(1)
    optimizer=torch.optim.Adam(head.parameters(),lr=.05)
    x=torch.tensor([[-2.],[-1.],[0.],[1.],[2.]]*10)
    y=torch.tensor([1,2,3,4,5]*10)
    before=float(models.loss_rows(head,x,y).mean().detach())
    for _ in range(100):
        optimizer.zero_grad(); loss=models.loss_rows(head,x,y).mean();loss.backward();optimizer.step()
    assert float(models.loss_rows(head,x,y).mean().detach())<before*.5
    predicted=head.probabilities(x).argmax(1)+1
    assert predicted[0]==1 and predicted[4]==5


class Tokenizer:
    def __call__(self,texts,**kwargs): return {'input_ids':[[int(v) for v in text.split()] for text in texts]}
    def num_special_tokens_to_add(self,pair=False): return 2
    def build_inputs_with_special_tokens(self,row): return [101]+row+[102]


def test_long_review_windows_keep_ending_and_short_reviews_match(monkeypatch):
    monkeypatch.setitem(enc.CONFIG,'length',8)
    first,last,long=enc.windows(Tokenizer(),['1 2 3 4 5 6 7 8 9','1 2'])
    assert first[0]==[101,1,2,3,4,5,6,102]
    assert last[0]==[101,4,5,6,7,8,9,102]
    assert first[1]==last[1] and long.tolist()==[True,False]


def test_temperature_preserves_class_decisions_and_reports_coverage_honestly():
    p=np.array([[.8,.05,.05,.05,.05],[.1,.1,.1,.1,.6]])
    calibrated=models.temperature_probabilities(p,2.)
    assert np.array_equal(calibrated.argmax(1),p.argmax(1))
    result=models.confidence_metrics(np.array([1,2]),p,.7)
    assert result['coverage']==.5 and result['confident_accuracy']==1. and result['overall_accuracy']==.5
    with pytest.raises(ValueError,match='positive'): models.temperature_probabilities(p,0)


def test_calibration_and_threshold_folds_are_product_disjoint():
    rows=[{'rating':i%5+1,'product':f'cal-product-{i//2}'} for i in range(600)]
    p=np.full((600,5),.02)
    p[np.arange(600),np.array([r['rating']-1 for r in rows])]=.92
    value=models.calibrate(rows,p)
    assert value['product_overlap']==0 and value['temperature_rows']+value['threshold_rows']==600
    assert value['threshold'] is not None and value['after_temperature_nll']<=value['before_temperature_nll']


def test_semantic_selection_keeps_control_when_gain_or_accuracy_guard_fails():
    baseline={'accuracy':.72,'macro_f1':.55,'nll':.8}
    results={sem.CONTROL:baseline,'ordered_ends':{'accuracy':.73,'macro_f1':.554,'nll':.7}}
    assert sem.selection(results)==sem.CONTROL
    results['ordered_ends'].update(macro_f1=.57,accuracy=.70)
    assert sem.selection(results)==sem.CONTROL
    results['ordered_ends']['accuracy']=.73
    assert sem.selection(results)=='ordered_ends'


def test_human_csv_displays_untrusted_text_without_executing_formulas():
    assert data.csv_display('=1+1')=="'=1+1"
    assert data.csv_display('  @SUM(1,2)')=="'  @SUM(1,2)"
    assert data.csv_display('good but expensive')=='good but expensive'


@pytest.fixture
def toy(tmp_path,monkeypatch):
    for k,v in {'fit_rows':50,'development_rows':50,'calibration_rows':300,'test_rows':50,'human_audit_per_rating':2}.items():
        monkeypatch.setitem(data.CONFIG,k,v)
    monkeypatch.setitem(enc.CONFIG,'dimension',8)
    monkeypatch.setitem(models.CONFIG,'head_epochs',3)
    monkeypatch.setitem(models.CONFIG,'head_minimum_epochs',2)
    monkeypatch.setitem(models.CONFIG,'fine_minimum_gain_over_lexical',2.)
    monkeypatch.setitem(sem.CONFIG,'bootstrap_product_repetitions',10)
    products={}
    for i in range(10000):
        p=f'semantic-fixture-{i}'
        products.setdefault(product_split(p),[]).append(p)
    words={1:'awful broken',2:'poor disappointing',3:'average adequate',4:'good useful',5:'excellent wonderful'}
    def row(y,split,p,text):
        value={'text':f'{words[y]} {text}','rating':y,'product':products[split][p]}
        value['id']=text_key(value['text']);return value
    fit=[row(y,'fit',0,f'fit demonstration {i}') for y in range(1,6) for i in range(10)]
    old_dev=[row(5,'development',0,'previous development sample')]
    dev=[row(i%5+1,'development',i+1,f'unseen development demonstration {i}') for i in range(1000)]
    old_test=[row(5,'test',0,'original consumed sample')]
    second=[row(4,'test',1,'second consumed sample')]
    third=[row(3,'test',2,'third consumed sample')]
    extra=[row(1,'test',i,f'never selected row on old product {i}') for i in range(3)]
    fresh=[row(i%5+1,'test',i+3,f'fresh test demonstration {i}') for i in range(150)]
    source=tmp_path/DIRECTORY/'source.jsonl';source.parent.mkdir(parents=True)
    source.write_text(''.join(json.dumps({'title':'','text':r['text'],'rating':r['rating'],'parent_asin':r['product']})+'\n'
        for r in fit+old_dev+dev+old_test+second+third+extra+fresh))
    ingest([source],source.parent/'audit.sqlite3')
    manifest={'audit_sha256':sha256_file(source.parent/'audit.sqlite3'),
        'available_by_partition_and_rating':{'fit':{str(y):10 for y in range(1,6)}}}
    write_json(source.parent/'manifest.json',manifest)
    directories={}
    for name,rows in [('v1',old_test),('v2',second),('v3',third)]:
        directory=tmp_path/'artifacts'/name;directory.mkdir(parents=True)
        path=directory/'fresh_check.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write_json(directory/'metadata.json',{'protocol':{'fresh_check':{'sha256':sha256_file(path)}}})
        write_json(directory/'final_experiment.json',{'status':'completed'})
        directories[name]=directory
    monkeypatch.setattr(data,'directory_for',lambda root:directories['v1'])
    monkeypatch.setattr(data.feedback_v2,'current_directory',lambda root:directories['v2'])
    monkeypatch.setattr(data.feedback_v3,'current_directory',lambda root:directories['v3'])
    monkeypatch.setattr(data,'load_partition',lambda root,name:{'development':old_dev,'test':old_test}[name])
    monkeypatch.setattr(sem.feedback_v3,'checked',lambda root:(directories['v3'],{}))
    monkeypatch.setattr(sem.feedback_v2,'checked',lambda root:(directories['v2'],{}))
    monkeypatch.setattr(sem,'binding',lambda root:{'fixture':True,'config':copy.deepcopy(sem.CONFIG)})
    model_dir=tmp_path/'models/feedback_minilm';model_dir.mkdir(parents=True)
    (model_dir/'config.json').write_text('fixture')
    encoder_manifest={'sha256':{'config.json':sha256_file(model_dir/'config.json')}}
    write_json(model_dir/'manifest.json',encoder_manifest)
    def fake_features(texts):
        features=np.zeros((len(texts),8),dtype=np.float32)
        for i,text in enumerate(texts):
            rating=next(y for y,phrase in words.items() if phrase in text)
            features[i,rating-1]=1.
        return features
    def build(root,directory,names,receipt):
        for name in names:
            if name=='test': assert read_json(directory/'final_experiment.json')['status']=='started'
            rows=data.load(directory,name,receipt)
            for view in enc.CONFIG['views']: np.save(directory/(name+'_'+view+'.npy'),fake_features([r['text'] for r in rows]))
            write_json(directory/(name+'_features.json'),{'offset':len(rows),'identity':{'config':enc.CONFIG},
                'sha256':{v:sha256_file(directory/(name+'_'+v+'.npy')) for v in enc.CONFIG['views']}})
        return encoder_manifest
    monkeypatch.setattr(sem,'build_features',build)
    monkeypatch.setattr(sem,'control_probabilities',lambda root,rows:np.full((len(rows),5),.2))
    monkeypatch.setattr(enc,'load_encoder',lambda root:(None,None,encoder_manifest))
    monkeypatch.setattr(enc,'embed_batch',lambda tokenizer,net,texts:({v:fake_features(texts) for v in enc.CONFIG['views']},0))
    return tmp_path,directories,old_dev,old_test+second+third


def test_new_development_calibration_test_reserve_entire_unused_products_and_blank_audit(toy):
    root,_,old_dev,old_test=toy
    directory=sem.prepare(root)
    receipt=read_json(directory/'protocol.json')['data']
    rows={n:data.load(directory,n,receipt) for n in ('fit','development','calibration','test')}
    assert receipt['product_overlap']==0 and receipt['previous_test_products']==3
    assert not {r['product'] for r in rows['development']+rows['calibration']} & {r['product'] for r in old_dev}
    assert not {r['product'] for r in rows['development']} & {r['product'] for r in rows['calibration']}
    assert not {r['product'] for r in rows['test']} & {r['product'] for r in old_test}
    assert read_json(directory/'human_audit_status.json')['completed_human_rows']==0


def test_actual_toy_models_reload_and_final_check_is_only_exposed_once(toy):
    root,previous,*_=toy
    before={str(p):sha256_file(p) for d in previous.values() for p in d.iterdir()}
    metadata=sem.train(root)
    assert metadata['test_exposed'] is False and metadata['fine_tuning']['status']=='skipped_development_guard'
    assert len(metadata['development'])==8 and metadata['selected_classifier']!=sem.CONTROL
    loaded=sem.SemanticModel(root)
    p=loaded.probabilities(['awful broken fictional product','excellent wonderful fictional item'])
    assert p.shape==(2,5) and np.array_equal(p.argmax(1)+1,[1,5])
    report=sem.evaluate(root)
    assert sem.evaluate(root)==report and report['selection_unchanged_after_test']
    assert report['selected_classifier']==metadata['selected_classifier']
    assert before=={str(p):sha256_file(p) for d in previous.values() for p in d.iterdir()}
    path=sem.current_directory(root)/'test.jsonl';path.write_text('changed')
    with pytest.raises(ValueError,match='final sample changed'):sem.evaluate(root)


def test_source_change_refuses_semantic_resume(toy,monkeypatch):
    root,*_=toy
    sem.prepare(root)
    monkeypatch.setattr(sem,'binding',lambda root:{'changed':True})
    with pytest.raises(ValueError,match='source/data/config changed'):sem.prepare(root)


def test_feature_checkpoint_resumes_unfinished_rows_and_rejects_tampering(tmp_path,monkeypatch):
    monkeypatch.setitem(enc.CONFIG,'dimension',4)
    monkeypatch.setitem(enc.CONFIG,'batch_size',2)
    rows=[{'text':str(i)} for i in range(6)]
    (tmp_path/'fit.jsonl').write_text('fixture features')
    calls=[]
    def embedded(tokenizer,encoder,texts):
        calls.append(texts)
        if len(calls)==2:raise RuntimeError('fixture interruption')
        x=np.eye(4,dtype=np.float32)[[int(t)%4 for t in texts]]
        return {'prefix':x,'ends':x},0
    monkeypatch.setattr(enc,'embed_batch',embedded)
    with pytest.raises(RuntimeError,match='fixture interruption'):
        enc.embed_rows(tmp_path,'fit',rows,None,None,{'fixture':True})
    assert read_json(tmp_path/'fit_features.json')['offset']==2
    arrays,receipt=enc.embed_rows(tmp_path,'fit',rows,None,None,{'fixture':True})
    assert calls[2]==['2','3'] and receipt['offset']==6
    assert np.array_equal(arrays['prefix'],np.eye(4,dtype=np.float32)[[0,1,2,3,0,1]])
    del arrays
    with (tmp_path/'fit_prefix.npy').open('r+b') as stream:stream.seek(-4,2);stream.write(b'xxxx')
    with pytest.raises(ValueError,match='checkpoint changed'):
        enc.embed_rows(tmp_path,'fit',rows,None,None,{'fixture':True})


def test_changed_encoder_bytes_refuse_completed_model_reload(toy):
    root,*_=toy
    sem.train(root)
    (root/'models/feedback_minilm/config.json').write_text('changed encoder')
    with pytest.raises(ValueError,match='encoder weights/tokenizer changed'):sem.checked(root)


def test_parallel_reservation_is_excluded_without_changing_protected_semantic_cohort(toy):
    root,_,_,old_test=toy
    first=sem.prepare(root)
    original=data.load(first,'test',read_json(first/'protocol.json')['data'])
    with data.connect(root) as db:
        data.block_table(db,{r['product'] for r in old_test})
        remaining=data.query_rows(db,'test',150)[50:100]
    parallel=root/'artifacts/feedback_v4_20261014T120000Z';parallel.mkdir()
    path=parallel/'fresh_check.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in remaining))
    write_json(parallel/'protocol.json',{'fresh_check':{'sha256':sha256_file(path)}})
    write_json(root/'artifacts/feedback_v4_latest.json',{'run_id':parallel.name})
    directory=root/'extra_reservation';directory.mkdir()
    receipt=data.reserve(root,directory)
    assert receipt['concurrent_product_overlap']==0
    assert data.load(directory,'test',receipt)==original
    assert receipt['concurrent_reservation']['products']==len(remaining)


def test_queue_lock_interoperates_with_existing_training_lock(tmp_path):
    from intentdesk.feedback_lab import training_lock
    path=Path(__file__).parents[1]/'scripts/run_feedback_semantic_job.py'
    spec=importlib.util.spec_from_file_location('semantic_runner_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    local=tmp_path/'local';local.mkdir(exist_ok=True)
    status={'state':'starting'}
    with training_lock(tmp_path):
        with pytest.raises(ValueError,match='training lock'):
            with module.shared_slot(tmp_path,False,status,local/'status.json',local):pass
    with module.shared_slot(tmp_path,False,status,local/'status.json',local):
        with pytest.raises(ValueError,match='already running'):
            with training_lock(tmp_path):pass


def test_loading_waiting_runner_does_not_load_torch_or_transformers():
    path=Path(__file__).parents[1]/'scripts/run_feedback_semantic_job.py'
    code="import runpy,sys;runpy.run_path(sys.argv[1],run_name='runner_fixture');assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_partial_finetune_updates_only_last_encoder_block(tmp_path,monkeypatch):
    monkeypatch.setitem(enc.CONFIG,'dimension',8)
    monkeypatch.setitem(models.CONFIG,'fine_epochs',1)
    monkeypatch.setitem(models.CONFIG,'minimum_free_memory_bytes',0)
    monkeypatch.setitem(models.CONFIG,'profile_max_rss_bytes',10**12)
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Module();self.encoder.layer=nn.ModuleList([nn.Linear(8,8),nn.Linear(8,8)])
        def forward(self,x):return self.encoder.layer[1](self.encoder.layer[0](x))
    torch.manual_seed(123)
    network=TinyEncoder();network.requires_grad_(False)
    before={k:v.clone() for k,v in network.state_dict().items()}
    monkeypatch.setattr(models,'load_encoder',lambda root:(None,network,{}))
    rows=[{'text':str(i%5),'rating':i%5+1} for i in range(20)]
    monkeypatch.setattr(models,'fine_features',lambda tok,net,texts,view:net(torch.nn.functional.one_hot(torch.tensor([int(t) for t in texts]),8).float()))
    models.save_torch(tmp_path/'nominal_prefix.pt',{'head':models.NominalHead(8).state_dict(),'kind':'nominal'})
    result=models.finetune(tmp_path,tmp_path,'nominal_prefix','nominal','prefix',rows,rows)
    assert result['completed_epochs']==1 and result['status']=='completed'
    assert torch.equal(network.encoder.layer[0].weight,before['encoder.layer.0.weight'])
    assert not torch.equal(network.encoder.layer[1].weight,before['encoder.layer.1.weight'])
    assert not network.encoder.layer[0].weight.requires_grad and network.encoder.layer[1].weight.requires_grad

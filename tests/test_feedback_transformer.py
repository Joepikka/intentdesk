import numpy as np
import pytest

torch = pytest.importorskip("torch")
from intentdesk.feedback_transformer import CONFIG, MultiHeadAttention, ScratchTransformer, tokenize, vocabulary


def test_vocabulary_is_fit_only_and_unknown_and_truncation_are_explicit():
    vocab = vocabulary(["good review", "good product"])
    encoded,truncated = tokenize(["unseen good", "good "*80],vocab,length=8)
    assert encoded.shape==(2,8) and encoded[0,0]==2 and encoded[0,1]==1
    assert truncated==1 and "unseen" not in vocab


def test_custom_attention_matches_uniform_attention_with_zero_queries():
    torch.manual_seed(4)
    attention = MultiHeadAttention(4,2,0.)
    with torch.no_grad():
        attention.qkv.weight.zero_(); attention.qkv.bias.zero_()
        attention.qkv.weight[8:12]=torch.eye(4)
        attention.output.weight.copy_(torch.eye(4)); attention.output.bias.zero_()
    x = torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.],[99.,99.,99.,99.]]])
    result = attention(x,torch.tensor([[True,True,False]]))
    assert torch.allclose(result,torch.tensor([[[3.,4.,5.,6.]]]).expand(1,3,4))


def test_scratch_transformer_backward_and_padding_invariance():
    torch.set_num_threads(2); torch.manual_seed(10)
    model = ScratchTransformer(40,16,4,2,length=12,dropout=0.).eval()
    short = torch.tensor([[2,3,4]])
    padded = torch.tensor([[2,3,4,0,0,0]])
    assert torch.allclose(model(short),model(padded),atol=1e-6)
    before = model.embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    loss = torch.nn.functional.cross_entropy(model(short),torch.tensor([3]))
    loss.backward()
    assert model.blocks[0].attention.qkv.weight.grad is not None
    assert model.embedding.weight.grad is not None and torch.isfinite(model.embedding.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before,model.embedding.weight)
    with pytest.raises(ValueError,match="CLS"):
        model(torch.zeros((1,4),dtype=torch.long))


def test_larger_model_really_has_more_trainable_parameters():
    small = ScratchTransformer(20000,**CONFIG["architectures"]["scratch_small"])
    large = ScratchTransformer(20000,**CONFIG["architectures"]["scratch_large"])
    assert sum(p.numel() for p in large.parameters()) > 2*sum(p.numel() for p in small.parameters())
    assert all(p.requires_grad for p in large.parameters())


def test_stopping_waits_for_twenty_epochs_but_honors_smaller_test_budget(monkeypatch):
    from intentdesk.feedback_transformer import can_stop
    assert not can_stop(19, 99)
    assert can_stop(20, CONFIG["patience"])
    assert not can_stop(20, 0)
    monkeypatch.setitem(CONFIG, "epochs", 1)
    assert can_stop(1, CONFIG["patience"])


def test_development_selection_does_not_confuse_majority_accuracy_with_macro_f1():
    from intentdesk.feedback_transformer import development_result
    class Majority(torch.nn.Module):
        def forward(self, ids):
            logits=torch.zeros((len(ids),5)); logits[:,4]=3
            return logits
    score,metrics=development_result(Majority(),torch.ones((4,2),dtype=torch.long),[5,5,5,1])
    assert metrics["development_accuracy"]==.75
    assert metrics["development_macro_f1"]==pytest.approx(.1714285714)
    assert score[0]==metrics["development_macro_f1"]


def test_old_loss_selected_checkpoint_continues_without_losing_history(tmp_path,monkeypatch):
    from intentdesk.feedback_transformer import fit_candidate
    from intentdesk.common import sha256_file,write_json
    monkeypatch.setitem(CONFIG,"architectures",{"scratch_small":{"dimension":8,"heads":2,"layers":1}})
    monkeypatch.setitem(CONFIG,"epochs",1)
    texts=[f"review rating example {i}" for i in range(10)]
    labels=[1,2,3,4,5]*2; vocab=vocabulary(texts)
    fit_candidate(tmp_path,"scratch_small",texts,labels,texts,labels,vocab)
    checkpoint=tmp_path/"scratch_small_resume.pt"
    state=torch.load(checkpoint,weights_only=True)
    state.pop("selection_version"); state.pop("best_score")
    for row in state["history"]:
        row.pop("development_macro_f1")
    torch.save(state,checkpoint)
    write_json(checkpoint.with_suffix(".json"),{"sha256":sha256_file(checkpoint)})
    monkeypatch.setitem(CONFIG,"epochs",2)
    model,trained=fit_candidate(tmp_path,"scratch_small",texts,labels,texts,labels,vocab)
    assert len(trained["history"])==2 and trained["continued_from_epochs"]==1
    assert all("development_macro_f1" in row for row in trained["history"])
    assert trained["best_epoch"] in (1,2)

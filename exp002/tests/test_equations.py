from dataclasses import replace
import numpy as np
import pytest
from exp002.model import TemporalField,Params,store2_rhs,shunt,habituate
from exp002.task import stream,OLD,LONG
from exp002.develop_probe import DEV_LONG
from exp002.controls import GRU,Readout,target


def test_store2_printed_equations_and_complementary_gating():
    x=np.array([.2,.1,0,0]);y=np.array([.18,.09,0,0]);v=np.array([0,0,1.,0])
    dx,dy=store2_rhs(x,y,v)
    np.testing.assert_allclose(dx,.01*v+y-x*x.sum()-.7*x)
    np.testing.assert_array_equal(dy,np.zeros(4))
    dx,dy=store2_rhs(x,y,np.zeros(4))
    np.testing.assert_array_equal(dx,np.zeros(4))
    np.testing.assert_allclose(dy,5*(x-y))


def test_low_rank_masking_equals_dense_source_coefficients():
    m=TemporalField(9)
    coefficients=m.size[None,:]*(1+m.support@m.support.T)
    coefficients/=coefficients.sum(axis=1,keepdims=True)
    signal=np.random.default_rng(1).uniform(size=len(m.size))
    np.testing.assert_allclose(m.masking(signal),coefficients@signal,atol=1e-14)
    assert m.size.max()==4
    assert len(m.size)==64


def test_shunting_and_habituation_derivatives_and_bounds():
    x=np.array([.1,.6]);ex=np.array([.8,.2]);inh=np.array([.3,.5]);dt=1e-7
    dy=(shunt(x,ex,inh,dt,lower=1,speed=2)-x)/dt
    rhs=2*(-.1*x+(1-x)*ex-(x+1)*inh)
    np.testing.assert_allclose(dy,rhs,atol=3e-7)
    for d in [.001,.1,10]:
        out=shunt(x,ex,inh,d,lower=1,speed=2)
        assert np.all((out>=-1)&(out<=1))
    z=np.array([.2,.8]);act=np.array([.1,.5])
    np.testing.assert_allclose((habituate(z,act,dt)-z)/dt,.01*(1-z)-z*(.1*act+3*act**2),atol=1e-7)


def test_feedback_cannot_create_unseen_wm_items_from_silence():
    m=TemporalField(9)
    m.c[:]=.9
    m.step(np.zeros(4),feedback=True)
    np.testing.assert_array_equal(m.x,np.zeros(4))
    np.testing.assert_array_equal(m.y,np.zeros(4))


def test_initial_order_is_not_given_a_correct_sequence_label():
    m=TemporalField(9);m2=TemporalField(9)
    np.testing.assert_array_equal(m.W,m2.W)
    assert set(LONG).isdisjoint(DEV_LONG)
    for support in np.unique(m.support,axis=0):
        w=m.W[(m.support==support).all(axis=1)]
        assert np.allclose(w.sum(axis=1),1)
        # Each candidate has same sorted supported weights, not a privileged order.
        np.testing.assert_allclose(np.sort(w,axis=1),np.broadcast_to(np.sort(w[0]),w.shape))


def test_learning_is_sensory_gated_and_no_chunk_ablation_keeps_lower_path():
    m=TemporalField(4);original=m.W.copy()
    m.run(stream('ABC',m.p,np.random.default_rng(1)),learn=True,chunking=False)
    np.testing.assert_array_equal(original,m.W)
    assert np.linalg.norm(m.x)>0
    m.run(stream('ABC',m.p,np.random.default_rng(1)),learn=True)
    assert np.linalg.norm(m.W-original)>0
    assert np.all(m.W[m.support==0]==0)


def test_time_order_changes_source_circuit_states():
    m=TemporalField(4)
    a=m.run(stream('ABC',m.p,np.random.default_rng(1)),feedback=False)[-1]
    b=m.run(stream('CBA',m.p,np.random.default_rng(1)),feedback=False)[-1]
    assert np.linalg.norm(a-b)>.001


def test_gru_gradients_not_a_deliberately_memoryless_control():
    m=GRU(2,hidden=3)
    a=np.random.default_rng(5).normal(size=(5,4))
    loss,g=m.loss_grad(a,'AB')
    for key,w in m.w.items():
        for idx in np.ndindex(w.shape):
            old=w[idx];eps=1e-5
            w[idx]=old+eps;plus=m.loss_grad(a,'AB')[0]
            w[idx]=old-eps;minus=m.loss_grad(a,'AB')[0]
            w[idx]=old
            assert g[key][idx]==pytest.approx((plus-minus)/(2*eps),abs=1e-7)
    assert np.all(m.w['bz']==2)


def test_rls_matches_independent_batch_ridge():
    rng=np.random.default_rng(5);x=rng.normal(size=(20,5));seqs=['A','AB','BC','ABC']*5
    h=Readout(5)
    for a,s in zip(x,seqs):h.update(a,s)
    xx=np.column_stack([x,np.ones(20)])
    yy=np.array([target(s) for s in seqs])
    batch=np.linalg.solve(xx.T@xx+.1*np.eye(6),xx.T@yy)
    np.testing.assert_allclose(h.B,batch,atol=1e-12)


def test_invalid_development_cannot_authorize_stability_claim():
    from exp002.readiness import assess,require_stage1_pass
    import json
    from pathlib import Path
    config=json.loads(Path('exp002/configs/readiness-v1.json').read_text())
    result=assess(config,{'headroom_calibrated':False,'numerical_max_difference':.005},
                  [{'distinct_winner_fraction':.2,'same_set_order_winner_discrimination':.72}])
    assert result['status']=='STOPPED_AT_DEVELOPMENT'
    assert result['stability_hypothesis_verdict']=='NOT_TESTED'
    assert result['stage2']==result['stage3']=='not_run'
    with pytest.raises(RuntimeError):require_stage1_pass(result)


def test_complete_readiness_still_is_not_stage1_validation():
    from exp002.readiness import assess,require_stage1_pass
    config={'maximum_step_refinement_relative_error':.02,'minimum_distinct_chunk_winner_fraction':.75,
            'minimum_same_set_order_discrimination':.8,'confirmation_difficulty':{'delay':2}}
    result=assess(config,{'headroom_calibrated':True,'numerical_max_difference':.005},
                  [{'distinct_winner_fraction':1.,'same_set_order_winner_discrimination':1.}])
    assert result['eligible_for_confirmation']
    with pytest.raises(RuntimeError):require_stage1_pass(result)

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


def test_full_cognitive_and_field_rhs_transcription():
    from exp002.model import positive_sigmoid
    p=Params(dt=1e-7)
    m=TemporalField(4,p)
    rng=np.random.default_rng(9)
    m.x=rng.uniform(.02,.15,4);m.y=rng.uniform(.05,.2,4)
    m.c=rng.uniform(-.1,.5,64);m.zi[:]=.8;m.zw[:]=.7;m.zc[:]=.6
    x,y,c=m.x.copy(),m.y.copy(),m.c.copy();v=np.array([0.,1.,0.,0.])
    td=8*(positive_sigmoid(c-.2)@m.M)
    ey=.1*v*.8+.7*x+(y>0)*td
    ex=.1*v*.8+.05*y
    iy=1.25*(ex.sum()-ex)
    expected_y=-.1*y+(1-y)*ey
    expected_x=-.1*x+(1-x)*ex-x*iy
    coeff=m.size[None,:]*(1+m.support@m.support.T)
    coeff/=coeff.sum(axis=1,keepdims=True)
    ec=70/m.size*(m.W@(x*.7))+m.size*positive_sigmoid(c*.6)
    ic=coeff@positive_sigmoid(c,.2)
    expected_c=2*(-.1*c+(1-c)*ec-(c+1)*ic)
    # Eliminate O(dt) finite-step derivative error by Richardson extrapolation.
    # This changes the numerical check, not model equations or experiment data.
    from copy import deepcopy
    half=deepcopy(m);half.p=replace(p,dt=p.dt/2)
    half.step(v);m.step(v)
    for old,full,fine,expected in [(y,m.y,half.y,expected_y),
                                   (x,m.x,half.x,expected_x),
                                   (c,m.c,half.c,expected_c)]:
        derivative=2*(fine-old)/(p.dt/2)-(full-old)/p.dt
        np.testing.assert_allclose(derivative,expected,atol=5e-8)


def test_physical_unit_adapter_recovers_listed_2011_weight_scales():
    from exp002.unit_consistency.model import PhysicalUnitsField
    m=PhysicalUnitsField(1)
    triple=np.where(m.size==3)[0][0]
    mask=np.where(m.support[triple]>0)[0]
    m.W[triple]=0;m.W[triple,mask]=[.5,1/3,1/6]
    physical=.1*m.size[triple]*m.W[triple,mask]
    np.testing.assert_allclose(physical,[.15,.1,.05])
    one=np.where(m.size==1)[0][0]
    assert (.1*m.size[one]*m.W[one]).sum()==pytest.approx(.1)


def test_adapter_matches_physical_currents_and_keeps_memories_normalized():
    from exp002.unit_consistency.model import PhysicalUnitsField
    m=PhysicalUnitsField(1);base=TemporalField(1)
    base.W=m.W*(.1*m.size[:,None]);base.M=m.M*(.1*m.size[:,None])
    original=m.W.copy()
    for _ in range(10):
        m.step(np.array([1.,0,0,0]),learn=False)
        base.step(np.array([1.,0,0,0]),learn=False)
    np.testing.assert_array_equal(m.state(),base.state())
    np.testing.assert_array_equal(m.W,original)
    m.step(np.array([1.,0,0,0]),learn=True)
    np.testing.assert_allclose(m.W.sum(axis=1),1,atol=1e-14)


def test_unit_variant_readiness_is_computed_not_a_fixed_negative_verdict():
    from exp002.unit_consistency.diagnose import assess_variant
    base={'performance':[{'condition':'test','field':.8,'frozen_filters':.5}],
          'shared_headroom_calibrated':True}
    mechanism=[{'trained_winners':20,'untrained_filter_winners':16}]
    result=assess_variant(base,mechanism,[{'maximum_relative_state_error':.005}])
    assert result['status']=='eligible_for_further_Stage1_validation'
    assert result['stage1_confirmation']=='not_run'
    blocked=assess_variant(base,mechanism,[{'maximum_relative_state_error':.04}])
    assert blocked['status']=='STOPPED_AT_DEVELOPMENT'

"""Executable no-go boundary; produces no drift conclusion from failed prerequisites."""
import argparse,json,hashlib,subprocess
from pathlib import Path


def assess(config,development,mechanisms):
    checks={
      'shared_difficulty_calibrated':bool(development['headroom_calibrated']),
      'integration_refinement':development['numerical_max_difference']<config['maximum_step_refinement_relative_error'],
      'distinct_chunk_winners':all(m['distinct_winner_fraction']>=config['minimum_distinct_chunk_winner_fraction'] for m in mechanisms),
      'same_set_order_discrimination':all(m['same_set_order_winner_discrimination']>=config['minimum_same_set_order_discrimination'] for m in mechanisms),
      'executable_confirmation_condition_frozen':config['confirmation_difficulty'] is not None}
    ready=all(checks.values())
    return {'experiment':'EXP002','status':'ready_for_stage1_confirmation' if ready else 'STOPPED_AT_DEVELOPMENT',
            'checks':checks,'eligible_for_confirmation':ready,
            'confirmation_seeds_executed':[],
            'stage1_confirmation':'not_run','stage2':'not_run','stage3':'not_run',
            'stability_hypothesis_verdict':'NOT_TESTED',
            'reason':'Readiness is not Stage 1 validation; no downstream interpretation is permitted.'}


def require_stage1_pass(result):
    if result.get('stage1_confirmation')!='passed':
        raise RuntimeError('Stage 1 prerequisites not confirmed; Stage 2/3 are prohibited')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='exp002/results/readiness-v1.json');args=p.parse_args()
    root=Path('exp002/results/development-calibration')
    config=json.loads(Path('exp002/configs/readiness-v1.json').read_text())
    result=assess(config,json.loads((root/'decision.json').read_text()),json.loads((root/'diagnostics/mechanisms.json').read_text()))
    result['freeze_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['config_sha256']=hashlib.sha256(Path('exp002/configs/readiness-v1.json').read_bytes()).hexdigest()
    out=Path(args.output)
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

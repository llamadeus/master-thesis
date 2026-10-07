#!/usr/bin/env python3
"""Export measured control cases and seed/fold forecast errors, without training.

Use --placeholder only to write empty inputs for the native Typst figures. Measured exports
require complete current-protocol control evaluations and scenario-level forecast
statistics. No values are interpolated or filled in for missing models.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from packages.trainer.evaluation.protocol import PROTOCOL_VERSION, PRIMARY_METRICS, scenario_average
from paper.tools.results_projection import forecast_projection

SEEDS=(17,42,123,1337,71026)


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def forecast_summary(rows):
    if len(rows)!=4000:raise ValueError("Expected 25 models, ten scenarios and sixteen horizon steps")
    models=[]
    for seed in SEEDS:
        for fold in range(5):
            selected=[r for r in rows if int(r['training_seed'])==seed and int(r['fold_id'])==fold]
            scenarios=sorted({r['scenario_id'] for r in selected})
            if len(scenarios)!=10:raise ValueError(f'Expected ten test scenarios for seed {seed}, fold {fold}')
            errors=[]
            for scenario in scenarios:
                values=[r for r in selected if r['scenario_id']==scenario]
                if sorted(int(r['horizon_step']) for r in values)!=list(range(1,17)):
                    raise ValueError('Expected one statistic per scenario and horizon step')
                count=sum(int(r['observation_count']) for r in values)
                if count<=0:raise ValueError('Empty forecast scenario')
                errors.append((sum(float(r['absolute_error_sum_mg_l']) for r in values)/count,
                    np.sqrt(sum(float(r['squared_error_sum_mg2_l2']) for r in values)/count)))
            models.append(dict(training_seed=seed,fold_id=fold,mae=float(np.mean([r[0] for r in errors])),rmse=float(np.mean([r[1] for r in errors]))))
    seed_means=[dict(training_seed=seed,**{metric:float(np.mean([m[metric] for m in models if m['training_seed']==seed])) for metric in ('mae','rmse')}) for seed in SEEDS]
    summary={metric:dict(mean=float(np.mean([s[metric] for s in seed_means])),sample_sd=float(np.std([s[metric] for s in seed_means],ddof=1))) for metric in ('mae','rmse')}
    return dict(data_kind='measured',models=models,seed_means=seed_means,summary=summary)


def forecast_rows(run):
    data=json.loads((run/'metrics.json').read_text())
    seed,fold=data.get('training_seed'),data.get('fold_id')
    if seed not in SEEDS or fold not in range(5) or data.get('scenario_weighting')!='equal':
        raise ValueError(f'Missing or incompatible seed/fold/scenario metadata: {run}')
    statistics=data['final']['test']['metrics']['chlorine']['scenario_horizon_statistics']
    return [dict(training_seed=seed,fold_id=fold,scenario_id=scenario,**row) for scenario,values in statistics.items() for row in values]


def control_rows(approach,seed,run,*,returns_only=False):
    """Read stored returns; legacy physical scores never leave returns-only mode."""
    if approach not in ('baseline','pretrained') or seed not in SEEDS:raise ValueError('Invalid control identity')
    if (run/'evaluation-incomplete.json').exists():raise ValueError(f'Incomplete evaluation: {run}')
    data=json.loads((run/'metrics.json').read_text())
    accepted_versions=(2,PROTOCOL_VERSION) if returns_only else (PROTOCOL_VERSION,)
    if data.get('protocol_version') not in accepted_versions or data.get('status')!='complete':raise ValueError(f'Legacy or incomplete evaluation: {run}')
    curve=[];cases=[]
    for checkpoint in data['history']:
        evaluation=checkpoint['evaluation']
        validate_inventory(evaluation['cases'],5)
        for nominal in checkpoint['nominal_thresholds']:
            actual=checkpoint['timesteps']
            if actual<nominal:raise ValueError('Checkpoint precedes its nominal threshold')
            curve.append(dict(approach=approach,training_seed=seed,nominal_interactions=nominal,actual_interactions=actual,
                evaluation_return=scenario_average(evaluation['cases'],lambda r:r['values']['evaluation_return'])))
            if not returns_only:
                cases.extend(_case_rows(evaluation,approach,seed,'validation',nominal,actual))
    if sorted(c['nominal_interactions'] for c in curve)!=list(range(0,1_000_001,50_000)):
        raise ValueError('Incomplete or duplicate learning-curve checkpoints')
    test=data.get('test')
    if not test or test.get('status')!='complete':raise ValueError('Missing completed final test')
    validate_inventory(test['cases'],10)
    if not returns_only:
        cases.extend(_case_rows(test,approach,seed,'test',1_000_000,data['final_timesteps']))
    return curve,cases


def validate_inventory(cases,schedules_per_scenario):
    grouped={}
    identities=set()
    for case in cases:
        identity=(case['scenario_id'],case['schedule_hash'])
        if identity in identities:raise ValueError('Duplicate evaluation case')
        identities.add(identity)
        grouped.setdefault(case['scenario_id'],set()).add(case['schedule_hash'])
    if len(grouped)!=10 or any(len(schedules)!=schedules_per_scenario for schedules in grouped.values()):
        raise ValueError(f'Expected ten scenarios with {schedules_per_scenario} schedules each')


def _case_rows(evaluation,approach,seed,split,nominal,actual):
    rows=[]
    for case in evaluation['cases']:
        for metric,value in (case['values']|case['components']).items():
            count=case['counts'].get(metric,{})
            rows.append(dict(approach=approach,training_seed=seed,split=split,nominal_interactions=nominal,
                actual_interactions=actual,scenario_id=case['scenario_id'],schedule_hash=case['schedule_hash'],
                metric=metric,value=value,applicable=value is not None,numerator=count.get('numerator'),
                denominator=count.get('denominator'),eligible_observations=count.get('eligible_observations',count.get('denominator')),clearance_time_seconds=case['clearance_time_seconds'],zero_injection_arsenic_burden=case['zero_injection_arsenic_burden']))
    return rows


def control_summary(rows):
    summaries=[]
    for metric in PRIMARY_METRICS:
        seed_values={approach:[] for approach in ('baseline','pretrained')}
        for approach in seed_values:
            for seed in SEEDS:
                cases=[row for row in rows if row['approach']==approach and row['training_seed']==seed and row['split']=='test' and row['metric']==metric]
                seed_values[approach].append(scenario_average(cases,lambda r:r['value']))
        paired=[p-b if b is not None and p is not None else None for b,p in zip(seed_values['baseline'],seed_values['pretrained'])]
        def summary(values):
            valid=[v for v in values if v is not None]
            return dict(seed_values=values,mean=float(np.mean(valid)) if valid else None,
                        sample_sd=float(np.std(valid,ddof=1)) if len(valid)>1 else None,
                        applicable_seeds=len(valid))
        # The design uses five independent training seeds. Missing seed-level
        # metrics remain NA rather than being silently treated as zero.
        interval=None
        if all(value is not None for value in paired):
            half=2.7764451051977987*np.std(paired,ddof=1)/np.sqrt(5)
            interval=[float(np.mean(paired)-half),float(np.mean(paired)+half)]
        summaries.append(dict(metric=metric,baseline=summary(seed_values['baseline']),
            pretrained=summary(seed_values['pretrained']),paired_difference=summary(paired),ci95=interval))
    return summaries


def validate_case_pairing(rows):
    for split in ('validation','test'):
        chosen=[r for r in rows if r['split']==split and r['metric']=='chlorine_ceiling_violation_fraction']
        groups={}
        for r in chosen:
            groups.setdefault((r['approach'],r['training_seed'],r['nominal_interactions']),{})[(r['scenario_id'],r['schedule_hash'])]=r['denominator']
        values=list(groups.values())
        if not values or any(v!=values[0] for v in values):
            raise ValueError('Evaluation cases or reference eligibility differ across checkpoints or policies')


def write_plot_data(path, rows=None):
    """Keep model/horizon errors and individual seed returns for native figures."""
    kind = 'pending' if rows is None else 'measured'
    data = (forecast_projection([] if rows is None else rows, kind)
            if path.name == 'forecast-plot-data.json'
            else dict(data_kind=kind, rows=[] if rows is None else rows))
    path.write_text(json.dumps(data,separators=(',', ':'),allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast-run',type=Path,action='append',default=[])
    parser.add_argument('--control-run',nargs=3,action='append',default=[],metavar=('APPROACH','SEED','DIRECTORY'))
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--placeholder',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.placeholder:
        if args.forecast_run or args.control_run:parser.error('Placeholder and measured inputs cannot be mixed')
        write_plot_data(out/'forecast-plot-data.json');write_plot_data(out/'learning-plot-data.json');return
    if not args.forecast_run and not args.control_run:parser.error('Supply measured runs or --placeholder')
    if args.forecast_run:
        rows=[row for run in args.forecast_run for row in forecast_rows(run)]
        summary=forecast_summary(rows)
        write_csv(out/'forecast_error_statistics.csv',rows);write_csv(out/'forecast_model_errors.csv',summary['models'])
        (out/'forecast_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        write_plot_data(out/'forecast-plot-data.json',rows)
    if args.control_run:
        curves=[];cases=[];identities=set()
        for approach,seed,run in args.control_run:
            identity=(approach,int(seed))
            if identity in identities:raise ValueError('Duplicate control run')
            identities.add(identity)
            c,r=control_rows(approach,int(seed),Path(run));curves.extend(c);cases.extend(r)
        if identities!={(a,s) for a in ('baseline','pretrained') for s in SEEDS}:raise ValueError('Both approaches and all five seeds are required')
        validate_case_pairing(cases)
        write_csv(out/'learning_curves.csv',curves);write_csv(out/'control_metrics.csv',cases)
        (out/'control_summary.json').write_text(json.dumps(control_summary(cases),indent=2)+'\n')
        write_plot_data(out/'learning-plot-data.json',curves)


if __name__=='__main__':main()

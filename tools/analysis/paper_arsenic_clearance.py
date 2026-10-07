"""Publication-only physical rescoring and clearance timing from saved final tests."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from packages.trainer.evaluation.protocol import reference_clusters, reference_windows, physical_metrics, THRESHOLD
from packages.trainer.forecasting.features import derive_transported_arsenic
from paper.tools.results_projection import SEEDS, scenario_metrics, CONTROL_COMPONENTS, PRIMARY_COMPONENTS

METRIC = "arsenic_clearance_lead"


def source_intervals(times, strengths):
    """Recover the fixed half-hour source schedule from lossless sampled strengths."""
    events = []
    for node in range(strengths.shape[1]):
        active = strengths[:, node] > 0
        changes = np.diff(np.pad(active.astype(int), (1, 1)))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
            if end == len(times):
                raise ValueError("Source injection reaches episode boundary")
            events.append(SimpleNamespace(start_time_seconds=float(times[start]),
                                          end_time_seconds=float(times[end])))
    return sorted(events, key=lambda e: e.start_time_seconds)


def clusters(times, transported, strengths):
    """Use the canonical threshold-only event protocol for publication timing."""
    return reference_clusters(times, transported, source_intervals(times, strengths))


def physical_projection(arrays):
    """Rescore saved concentrations/actions under the corrected reference regime."""
    rt = arrays['reference_time_seconds']
    events = source_intervals(rt, arrays['reference_event_strength_by_node'])
    event, post = reference_windows(rt, arrays['reference_transported_arsenic'], events)
    scored = physical_metrics(arrays['time_seconds'], arrays['chlorine'], arrays['transported_arsenic'],
                              arrays['injection'], event[1:], post[1:], 1800)
    return scored, event


def component_value(score, metric, component):
    if component == 'total' or metric in ('chlorine_shortfall_fraction', 'normal_chlorine_excess_fraction'):
        return score['values'][metric]
    return score['components'][f'{metric}_{component}']


def persistent_clearance(times, aggregate, start, stop):
    """First report after last exceedance; None denotes no observed exceedance."""
    selected = np.flatnonzero((times >= start) & (times < stop))
    above = selected[aggregate[selected] > THRESHOLD]
    if not len(above):
        return None
    last = int(above[-1])
    if last == selected[-1]:
        raise ValueError(f"Persistent clearance not observed in cluster [{start}, {stop})")
    return float(times[last + 1])


def load_case(path, record):
    with np.load(path / record['file'], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in (
            'time_seconds', 'arsenic', 'transported_arsenic', 'reference_time_seconds',
            'reference_arsenic', 'reference_transported_arsenic', 'reference_event',
            'reference_event_strength_by_node', 'reference_hydraulic_outflow', 'chlorine', 'injection')}
    for key in ('arsenic', 'transported_arsenic', 'chlorine', 'reference_arsenic', 'reference_transported_arsenic', 'reference_event_strength_by_node', 'reference_hydraulic_outflow'):
        expected_rows = 1200 if key in ('arsenic', 'transported_arsenic', 'chlorine') else 1201
        if arrays[key].shape != (expected_rows, 31) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid monitored-node shape or values: {path / record['file']}, {key}")
    if arrays['injection'].shape != (1200,) or not np.isfinite(arrays['injection']).all():
        raise ValueError(f"Invalid injection samples: {path / record['file']}")
    rt, pt = arrays['reference_time_seconds'], arrays['time_seconds']
    if len(rt) != 1201 or not np.array_equal(pt, rt[1:]) or not np.all(np.diff(rt) == 1800):
        raise ValueError(f"Invalid half-hour sampling: {path / record['file']}")
    for raw_key, transported_key in (('reference_arsenic', 'reference_transported_arsenic'), ('arsenic', 'transported_arsenic')):
        raw = arrays[raw_key] if raw_key.startswith('reference') else np.vstack((np.zeros((1, arrays['arsenic'].shape[1])), arrays['arsenic']))
        calculated, _ = derive_transported_arsenic(raw, arrays['reference_event_strength_by_node'],
            arrays['reference_hydraulic_outflow'], rt, flow_units=record['flow_units'])
        expected = arrays[transported_key] if raw_key.startswith('reference') else arrays[transported_key]
        actual = calculated if raw_key.startswith('reference') else calculated[1:]
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-9):
            raise ValueError(f"Transported arsenic reconstruction mismatch: {path / record['file']}")
        if not np.isfinite(expected).all() or np.any(expected < 0):
            raise ValueError(f"Invalid concentrations: {path / record['file']}")
    return arrays


def clearance_projection(root, run_paths):
    rows, supports, selected_events, selected_cases = [], [], [], []
    case_records = {}
    selected_burdens, selected_rows = [], []
    selected_identity = None
    reference_by_case = {}
    common_inventory = None
    for seed in SEEDS:
        plan = json.loads((root/'pipelines/hanoi-principal-v89'/f'seed-{seed}'/'plan.json').read_text())
        policy_cases, directories = {}, {}
        for approach in ('baseline', 'pretrained'):
            directory = run_paths(root, plan, approach)
            metrics = json.loads((directory/'metrics.json').read_text())
            published = json.loads((directory/'best/manifest.json').read_text())
            selected = max(sorted(metrics['history'], key=lambda row: row['timesteps']), key=lambda row: row['evaluation']['evaluation_return'])
            key = 'baseline' if approach == 'baseline' else 'mpc_teacher_bc'
            if published['evaluation'] != selected['evaluation'] or metrics['final_validation'] != selected['evaluation'] or published['job_hash'] != plan['control'][key]['job_hash'] or published['job']['training_seed'] != seed:
                raise ValueError(f"Invalid validation-selected policy: {directory}")
            directories[approach] = directory/'evaluation-data/final/test'
            manifest = json.loads((directories[approach]/'manifest.json').read_text())
            policy_cases[approach] = {(c['scenario_id'], c['schedule_hash']): c for c in manifest['cases']}
        if set(policy_cases['baseline']) != set(policy_cases['pretrained']) or len(policy_cases['baseline']) != 100:
            raise ValueError(f"Unpaired final test inventory: seed {seed}")
        inventory = set(policy_cases['baseline'])
        if common_inventory is not None and inventory != common_inventory:
            raise ValueError(f"Case identities differ across seeds: {seed}")
        common_inventory = inventory
        support = dict(seed=str(seed), reference_events=0, eligible_events=0,
                       baseline_no_exceedance=0, pretrained_no_exceedance=0, paired_exclusions=0,
                       applicable_cases=0, no_eligible_cases=0)
        for identity in sorted(policy_cases['baseline']):
            data = {}
            for a in directories:
                try:
                    data[a] = load_case(directories[a], policy_cases[a][identity])
                except ValueError as exc:
                    raise ValueError(f"seed={seed}, policy={a}, case={identity}: {exc}") from exc
            baseline = data['baseline']
            common = {key: baseline[key] for key in baseline if key.startswith('reference_')}
            if identity in reference_by_case:
                common_previous = reference_by_case[identity]
                if any(not np.array_equal(common[k], common_previous[k]) for k in common):
                    raise ValueError(f"Reference differs across seeds: {identity}")
            else:
                reference_by_case[identity] = common
            if any(not np.array_equal(common[k], data['pretrained'][k]) for k in common):
                raise ValueError(f"Reference differs across policies: {seed}, {identity}")
            try:
                intervals = clusters(common['reference_time_seconds'], common['reference_transported_arsenic'],
                                     common['reference_event_strength_by_node'])
            except ValueError as exc:
                raise ValueError(f"seed={seed}, reference, case={identity}: {exc}") from exc
            leads = defaultdict(list)
            event_details = []
            for index, interval in enumerate(intervals):
                start, stop = interval.source_start_seconds, interval.next_source_start_seconds
                clear = {}
                for a in ('reference', 'baseline', 'pretrained'):
                    times = common['reference_time_seconds'] if a == 'reference' else data[a]['time_seconds']
                    aggregate = common['reference_transported_arsenic'].sum(axis=1) if a == 'reference' else data[a]['transported_arsenic'].sum(axis=1)
                    try:
                        clear[a] = persistent_clearance(times, aggregate, start, stop)
                    except ValueError as exc:
                        raise ValueError(f"seed={seed}, policy={a}, case={identity}, cluster={index+1}: {exc}") from exc
                reference = clear.pop('reference')
                support['reference_events'] += 1
                for a in directories:
                    support[f'{a}_no_exceedance'] += clear[a] is None
                eligible = all(v is not None for v in clear.values())
                support['eligible_events' if eligible else 'paired_exclusions'] += 1
                event_details.append(dict(event=index+1, source_start_h=start/3600,
                    reference_clearance_h=reference/3600,
                    baseline_clearance_h=clear['baseline']/3600 if clear['baseline'] is not None else None,
                    pretrained_clearance_h=clear['pretrained']/3600 if clear['pretrained'] is not None else None,
                    eligible=eligible))
                if eligible:
                    for a in directories:
                        leads[a].append((reference-clear[a])/3600)
            support['applicable_cases' if leads['baseline'] else 'no_eligible_cases'] += 1
            scores = {}
            for a in directories:
                score, corrected_event = physical_projection(data[a])
                scores[a] = score['values'] | score['components']
                for metric, component in CONTROL_COMPONENTS:
                    if metric == METRIC:
                        value = float(np.mean(leads[a])) if leads[a] else None
                    else:
                        value = component_value(score, metric, component)
                    rows.append(dict(approach=a, seed=str(seed), scenario=identity[0], metric=metric,
                                     component=component, value=value))
            main_tail = (common['reference_time_seconds'] > 2*86400) & (common['reference_time_seconds'] <= 25*86400)
            reference_burden = float(.5 * common['reference_transported_arsenic'][corrected_event & main_tail].sum())
            record = policy_cases['baseline'][identity]
            case_record = case_records.setdefault(identity, dict(scenario_id=identity[0], schedule_id=f"arsenic-{record['index']//10}",
                                     reference=reference_burden, baseline={}, pretrained={}))
            if not np.isclose(case_record['reference'], reference_burden, rtol=0, atol=0):
                raise ValueError(f"Rescored reference burden differs: {identity}")
            for a in directories:
                case_record[a][str(seed)] = scores[a]['arsenic_burden']
            if identity[0] == 'scenario-90' and record['index']//10 == 8:
                selected_identity = identity
                selected_burdens.append(dict(seed=str(seed), reference=reference_burden,
                                             baseline=scores['baseline']['arsenic_burden'], pretrained=scores['pretrained']['arsenic_burden']))
                if seed == SEEDS[0]:
                    for a in directories:
                        for metric, component in PRIMARY_COMPONENTS:
                            value = float(np.mean(leads[a])) if metric == METRIC and leads[a] else None if metric == METRIC else scores[a][metric]
                            selected_rows.append(dict(approach=a, metric=metric, component=component, value=value))
                    selected_rows.append(dict(approach='zero_injection', metric='arsenic_burden', component='total', value=reference_burden))
                selected_events.append(dict(seed=str(seed), events=event_details))
                selected_cases.append(dict(seed=str(seed), baseline=float(np.mean(leads['baseline'])) if leads['baseline'] else None,
                                           pretrained=float(np.mean(leads['pretrained'])) if leads['pretrained'] else None))
        supports.append(support)
    if selected_identity is None:
        raise ValueError("Missing selected illustrative case")
    case_points = [dict(scenario_id=r['scenario_id'], schedule_id=r['schedule_id'],
                        baseline=float(np.mean(list(r['baseline'].values()))),
                        pretrained=float(np.mean(list(r['pretrained'].values())))) for _, r in sorted(case_records.items())]
    return dict(rows=scenario_metrics(rows), support=supports, selected_events=selected_events,
                selected_cases=selected_cases, selected_burdens=selected_burdens, selected_rows=selected_rows,
                reference_values=[r['reference'] for _,r in sorted(case_records.items())], case_burdens=case_points,
                selected_identity=list(selected_identity))

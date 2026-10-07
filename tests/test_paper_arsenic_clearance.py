import json
import os
from pathlib import Path
import unittest
import numpy as np
from tools.analysis.paper_arsenic_clearance import persistent_clearance, clusters, physical_projection, load_case, component_value
from packages.trainer.evaluation.protocol import reference_windows
from types import SimpleNamespace


class PaperClearanceTests(unittest.TestCase):
    def test_persistent_time_ignores_temporary_dips_and_source_stop(self):
        times = np.arange(7)*1800.
        self.assertEqual(persistent_clearance(times, np.array([0,.02,0,.03,0,0,0]),0,10801),7200.)
        self.assertIsNone(persistent_clearance(times,np.zeros(7),0,10801))
        with self.assertRaisesRegex(ValueError,'not observed'):
            persistent_clearance(times,np.array([0,0,0,0,0,0,.02]),0,10801)

    def test_threshold_clusters_split_despite_overlapping_sources(self):
        times=np.arange(12)*1800.
        strengths=np.zeros((12,2)); strengths[1:5,0]=1; strengths[4:6,1]=2; strengths[9:10,0]=1
        transported=np.array([0,.02,0,0,.02,0,0,0,0,.02,0,0])[:,None]
        found=clusters(times,transported,strengths)
        self.assertEqual([(c.source_start_seconds,c.next_source_start_seconds,c.clearance_time_seconds) for c in found],
                         [(1800.,7200.,3600.),(7200.,16200.,9000.),(16200.,21600.,18000.)])
        events=[SimpleNamespace(start_time_seconds=1800,end_time_seconds=9000),
                SimpleNamespace(start_time_seconds=7200,end_time_seconds=10800),
                SimpleNamespace(start_time_seconds=16200,end_time_seconds=18000)]
        mask,_=reference_windows(times,transported,events)
        np.testing.assert_array_equal(np.flatnonzero(mask),[1,4,9])

    def test_physical_projection_uses_corrected_mask_not_saved_legacy_mask(self):
        times=np.arange(1201)*1800.
        strengths=np.zeros((1201,1));strengths[100:110]=1
        reference=np.zeros((1201,1));reference[101:103]=.02
        arrays=dict(reference_time_seconds=times,time_seconds=times[1:],
                    reference_event_strength_by_node=strengths,reference_transported_arsenic=reference,
                    reference_event=np.ones(1201,dtype=bool),chlorine=np.full((1200,1),.5),
                    transported_arsenic=reference[1:],injection=np.ones(1200))
        scores,mask=physical_projection(arrays)
        self.assertEqual(np.flatnonzero(mask).tolist(),[101,102])
        self.assertEqual(scores['counts']['normal_chlorine_excess_fraction']['denominator'],1006)
        self.assertAlmostEqual(scores['values']['arsenic_burden'],.02)
        self.assertEqual(scores['components']['chlorine_use_main_event'],60.)
        self.assertEqual(component_value(scores,'chlorine_use','main_normal'),30180.)
        self.assertEqual(component_value(scores,'chlorine_use','total'),30240.)


class SavedPrincipalReprojectionTests(unittest.TestCase):
    root=Path(os.environ.get('THESIS_RESULTS_ROOT', 'data'))

    @classmethod
    def setUpClass(cls):
        plan_path=cls.root/'pipelines/hanoi-principal-v89/seed-17/plan.json'
        if not plan_path.is_file():
            raise unittest.SkipTest('External saved Principal test artifacts unavailable')
        from tools.analysis.export_principal_v89_paper_results import run_paths
        cls.run_directory=run_paths(cls.root,json.loads(plan_path.read_text()),'baseline')
        cls.directory=cls.run_directory/'evaluation-data/final/test'
        cls.manifest=json.loads((cls.directory/'manifest.json').read_text())

    def test_real_case_10_and_76_reference_events_end_before_source_shutdown(self):
        for index in (10,76):
            with self.subTest(index=index):
                arrays=load_case(self.directory,self.manifest['cases'][index])
                _,corrected=physical_projection(arrays)
                self.assertGreater(np.count_nonzero(arrays['reference_event'] & ~corrected),0)
                active_sources=arrays['reference_event_strength_by_node'].sum(axis=1)>0
                self.assertTrue(np.any(active_sources & arrays['reference_event'] & ~corrected))
                found=clusters(arrays['reference_time_seconds'],arrays['reference_transported_arsenic'],arrays['reference_event_strength_by_node'])
                aggregate=arrays['reference_transported_arsenic'].sum(axis=1)
                for c in found:
                    self.assertEqual(c.clearance_time_seconds,persistent_clearance(arrays['reference_time_seconds'],aggregate,c.source_start_seconds,c.next_source_start_seconds))

    def test_legacy_scores_are_rejected_and_returns_only_contains_no_physical_rows(self):
        from tools.analysis.export_evaluation_results import control_rows
        with self.assertRaisesRegex(ValueError,'Legacy or incomplete'):
            control_rows('baseline',17,self.run_directory)
        returns,physical=control_rows('baseline',17,self.run_directory,returns_only=True)
        self.assertEqual(len(returns),21)
        self.assertEqual(physical,[])
        existing=json.loads(Path('paper/results/learning-plot-data.json').read_text())['rows']
        expected=[r for r in existing if r['approach']=='baseline' and r['training_seed']==17]
        self.assertEqual(returns,expected)


if __name__=='__main__':unittest.main()

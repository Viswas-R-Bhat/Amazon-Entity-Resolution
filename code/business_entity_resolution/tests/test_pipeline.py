import csv
import tempfile
import unittest
import zipfile
from unittest import mock
from pathlib import Path
import numpy as np
from er.features import FEATURES, features
from er.index import build_index, Retriever
from er.inference import predict, validate, MATCH_HEADER, CANDIDATE_HEADER
from er.io import HEADER, dump, digest, rows
from er.index import RETRIEVAL_VERSION
from er.metrics import entity_score, select_threshold, evaluate
from er.packaging import package
from er.text import normalize, unicode_text, record, fold


def raw(eid, name, address='', country='France'):
    return dict(zip(HEADER, (eid, name, address, country)))


def write(path, header, data):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data)


class MetricsTests(unittest.TestCase):
    def test_report_matches_independent_integer_formula(self):
        rng = np.random.default_rng(123)
        groups, scores, expected = [], [], []
        for i in range(50):
            truth = {f'S2-{j}' for j in range(int(rng.integers(0, 8)))}
            candidates = [f'S2-{j}' for j in range(int(rng.integers(0, 10)))]
            p = rng.random(len(candidates))
            prediction = {eid for eid, probability in zip(candidates, p) if probability >= .64}
            tp, fp, fn = len(truth & prediction), len(prediction-truth), len(truth-prediction)
            expected.append(1 if not truth and not prediction else 5*tp/(5*tp+4*fp+fn))
            start = len(scores)
            scores.extend(p)
            groups.append(dict(eid=f'S1-{i}',country='fixture',truth=list(truth),
                               candidates=candidates,start=start,end=len(scores)))
        self.assertAlmostEqual(evaluate(groups, np.asarray(scores), .64)['macro_f05'],
                               sum(expected)/len(expected), places=14)

    def test_official_empty_and_multimatch_semantics(self):
        self.assertEqual(entity_score([], []), 1)
        self.assertEqual(entity_score([], ['a']), 0)
        self.assertEqual(entity_score(['a'], []), 0)
        self.assertAlmostEqual(entity_score(['a', 'b'], ['a', 'c']), .5)
        self.assertAlmostEqual(entity_score(['a', 'b'], ['a']), 1.25 / 1.5)

    def test_zero_candidate_entities_stay_in_denominator(self):
        groups = [{'eid':'S1-1', 'country':'France', 'truth':['S2-1'], 'candidates':[], 'start':0,'end':0},
                  {'eid':'S1-2', 'country':'France', 'truth':[], 'candidates':[], 'start':0,'end':0}]
        report = evaluate(groups, [], .5)
        self.assertEqual(report['macro_f05'], .5)
        self.assertEqual(report['candidate_oracle_macro_f05'], .5)
        self.assertEqual(select_threshold(groups, [])['macro_f05'], .5)

    def test_threshold_optimizes_entities_not_pair_accuracy(self):
        groups = [{'eid':'S1-1', 'country':'India', 'truth':['S2-1'], 'candidates':['S2-1','S3-1'], 'start':0,'end':2},
                  {'eid':'S1-2', 'country':'India', 'truth':[], 'candidates':['S2-2'], 'start':2,'end':3}]
        scores = np.array([.9, .4, .8])
        selection = select_threshold(groups, scores)
        self.assertAlmostEqual(selection['macro_f05'], 1)
        self.assertGreater(selection['threshold'], .8)
        self.assertLessEqual(selection['threshold'], .9)


class TextTests(unittest.TestCase):
    def test_normalization_and_unseen_country(self):
        self.assertEqual(normalize('Café & Fils, SARL'), 'cafe and fils sarl')
        self.assertIn('\u093f', unicode_text('किरण'))
        self.assertTrue(normalize('किरण'))
        self.assertEqual(record(raw('S1-1','x',country='France'))['country'],
                         record(raw('S2-1','x',country=' france '))['country'])

    def test_missing_fields_are_not_perfect_matches(self):
        query = record(raw('S1-1','A'))
        target = record(raw('S2-1','A'))
        result = dict(zip(FEATURES, features(query, target, {'name_frequency':1, 'retrieval_score':0, 'exact_block':1})))
        self.assertEqual(result['address_equal'], 0)
        self.assertEqual(result['address_ratio'], 0)
        self.assertEqual(result['phonetic_ratio'], 0)
        self.assertEqual(result['address_missing'], 1)

    def test_group_fold_is_deterministic(self):
        self.assertEqual(fold('S1-123'), fold('S1-123'))
        self.assertEqual({fold(f'S1-{i}') for i in range(100)}, {'fit','tune','eval'})


class DiskPipelineTests(unittest.TestCase):
    def test_native_rank_preserves_cutoff_ties_and_country(self):
        target_path = self.test / 'ties.tsv'
        data = [raw(f'S2-{i:04}', 'Repeated Cafe', '12 Main Street',
                    'India' if i % 3 == 0 else 'France') for i in range(300)]
        write(target_path, HEADER, [r.values() for r in data])
        index = self.work / 'ties.sqlite'
        build_index([target_path], index)
        retriever = Retriever(index, per_pass=11)
        try:
            country = record(raw('S1-x', 'Repeated Cafe'))['country']
            expression = 'name:"repeated"'
            expected = retriever.db.execute(
                "SELECT search.rowid FROM search CROSS JOIN records AS r ON r.rid=search.rowid "
                "WHERE search MATCH ? AND r.country=? "
                "ORDER BY bm25(search,3.0,1.0,2.0,0.0),search.rowid LIMIT ?",
                (expression, country, retriever.per_pass)).fetchall()
            actual = retriever.ranked_hits(expression, country)
            self.assertEqual([r[0] for r in actual], [r[0] for r in expected])
            self.assertEqual(retriever.ranked_hits('name:"absent"', country), [])
        finally:
            retriever.close()

    def test_parallel_retrieval_matches_serial_in_order(self):
        from er.training import retrieve_rows
        selected = [raw('S1-a', 'Cafe Paris', '12 rue nord'),
                    raw('S1-b', 'Unknown', '', 'India'),
                    raw('S1-c', 'Cafe Paris', '12 rue nord')] * 6
        path = self.work / 'test.sqlite'
        retriever = Retriever(path)
        try:
            serial = list(retrieve_rows(selected, retriever, path, 60, 1))
            parallel = list(retrieve_rows(selected, retriever, path, 60, 2))
            self.assertEqual(serial, parallel)
        finally:
            retriever.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'dataset'
        self.test = self.dataset / 'test'
        self.test.mkdir(parents=True)
        self.work = self.root / 'work'
        self.work.mkdir()
        self.output = self.root / 'output'
        self.output.mkdir()
        self.references = [raw('S1-1', 'Café Lumiere', '12 Rue Paris'), raw('S1-2','Unique No Match','Zanzibar')]
        targets = [raw('S2-1','Cafe Lumiere SARL','12 Rue Paris'),
                   raw('S2-2','Cafe Lumiere','99 Washington Road','US'),
                   raw('S2-3','Other Ventures','40 Pine Road')]
        more = [raw('S3-1','Lumiere Cafe','Rue Paris 12'), raw('S3-2','Noise Enterprise','77 Manhattan')]
        for s, data in enumerate([self.references, targets, more], 1):
            write(self.test / f'test_source{s}.tsv', HEADER, [r.values() for r in data])
        self.paths = [self.test / f'test_source{s}.tsv' for s in (2,3)]
        build_index(self.paths, self.work / 'test.sqlite', batch_size=2)

    def tearDown(self):
        self.tmp.cleanup()

    def outputs(self, matches='S2-1,S3-1', candidates='S2-1,S3-1'):
        write(self.output / 'matching_results.tsv', MATCH_HEADER, [('S1-1',matches),('S1-2','')])
        write(self.output / 'candidate_pairs.tsv', CANDIDATE_HEADER, [('S1-1',candidates),('S1-2','')])

    def test_multimatch_country_and_cap(self):
        retriever = Retriever(self.work / 'test.sqlite', cap=2)
        ids = {r['eid'] for r, _ in retriever.candidates(record(self.references[0]))}
        retriever.close()
        self.assertEqual(ids, {'S2-1','S3-1'})

    def test_idempotent_completed_index(self):
        build_index(self.paths, self.work / 'test.sqlite')
        retriever = Retriever(self.work / 'test.sqlite')
        self.assertEqual(retriever.db.execute('SELECT count(*) FROM records').fetchone()[0], 5)
        retriever.close()

    def test_changed_input_fails_closed(self):
        with self.paths[0].open('a', encoding='utf-8') as handle:
            handle.write('S2-99\tTest\t\tFrance\n')
        with self.assertRaises(ValueError):
            build_index(self.paths, self.work / 'test.sqlite')

    def test_strict_valid_output(self):
        self.outputs()
        self.assertEqual(validate(self.dataset, self.work, self.output)['entities'], 2)

    def test_matches_outside_candidates_fail(self):
        self.outputs(candidates='S2-1')
        with self.assertRaises(ValueError):
            validate(self.dataset, self.work, self.output)

    def test_unknown_target_fails(self):
        self.outputs(matches='S2-999', candidates='S2-999')
        with self.assertRaises(ValueError):
            validate(self.dataset, self.work, self.output)

    def test_duplicate_target_fails(self):
        self.outputs(candidates='S2-1,S2-1,S3-1')
        with self.assertRaises(ValueError):
            validate(self.dataset, self.work, self.output)

    def test_missing_reference_fails(self):
        self.outputs()
        write(self.output / 'matching_results.tsv', MATCH_HEADER, [('S1-1','S2-1')])
        with self.assertRaises(ValueError):
            validate(self.dataset, self.work, self.output)

    def test_index_resumes_committed_batches(self):
        interrupted_index = self.work / 'interrupted.sqlite'
        def interrupted(path):
            for i, row in enumerate(rows(path)):
                if i == 2:
                    raise RuntimeError('Simulated interruption')
                yield row
        with mock.patch('er.index.rows', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                build_index(self.paths, interrupted_index, batch_size=2)
        build_index(self.paths, interrupted_index, batch_size=2)
        retriever = Retriever(interrupted_index)
        self.assertEqual(retriever.db.execute('SELECT count(*) FROM records').fetchone()[0], 5)
        self.assertEqual(len(retriever.candidates(record(self.references[0]))), 2)
        retriever.close()

    def fake_model(self):
        (self.work / 'model.txt').write_text('fixture', encoding='utf-8')
        dump(self.work / 'model_config.json', {'threshold':.5, 'cap':60, 'features':FEATURES,
             'retrieval_version':RETRIEVAL_VERSION, 'model_sha256':digest(self.work / 'model.txt')})
        model = mock.Mock()
        model.predict.side_effect = lambda values, **kwargs: np.ones(len(values))
        return model

    def test_prediction_checkpoint_reuse_and_partial_resume(self):
        with mock.patch('er.inference.lgb.Booster', return_value=self.fake_model()):
            predict(self.dataset, self.work, self.output, chunk_size=1)
            before = digest(self.output / 'matching_results.tsv')
            with mock.patch.object(Retriever, 'candidates', side_effect=AssertionError('Must reuse checkpoints')):
                predict(self.dataset, self.work, self.output, chunk_size=1)
            self.assertEqual(before, digest(self.output / 'matching_results.tsv'))
            (self.output / '.checkpoints/0000002.json').unlink()
            predict(self.dataset, self.work, self.output, chunk_size=1)
            self.assertEqual(before, digest(self.output / 'matching_results.tsv'))

    def test_corrupt_checkpoint_is_rejected(self):
        with mock.patch('er.inference.lgb.Booster', return_value=self.fake_model()):
            predict(self.dataset, self.work, self.output, chunk_size=1)
            (self.output / '.checkpoints/0000001.candidates').write_text('corrupt', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Corrupt checkpoint'):
                predict(self.dataset, self.work, self.output, chunk_size=1)

    def test_parallel_prediction_outputs_equal_serial(self):
        serial = self.root / 'serial'
        parallel = self.root / 'parallel'
        with mock.patch('er.inference.lgb.Booster', return_value=self.fake_model()):
            predict(self.dataset, self.work, serial, threads=1, chunk_size=1)
            predict(self.dataset, self.work, parallel, threads=2, chunk_size=1)
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            self.assertEqual((serial / name).read_bytes(), (parallel / name).read_bytes())

    def test_package_structure(self):
        self.fake_model()
        config = {'threshold':.5, 'cap':60, 'trees':1, 'model_sha256':digest(self.work / 'model.txt')}
        dump(self.work / 'model_config.json', config)
        self.outputs()
        groups = [{'eid':'S1-1', 'country':'France', 'truth':['S2-1','S3-1'],
                   'candidates':['S2-1','S3-1'], 'start':0, 'end':2}]
        evaluation = evaluate(groups, np.ones(2), .5)
        dump(self.work / 'validation_report.json', {'sample_entities':1, 'folds':{'eval':evaluation}})
        dump(self.output / 'run_summary.json', {'model_sha256':config['model_sha256']})
        archive = self.root / 'fixture.zip'
        package(self.dataset, self.work, self.output, 'Fixture Team', 'Fixture Member', archive)
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            self.assertIn('output/matching_results.tsv', names)
            self.assertIn('output/candidate_pairs.tsv', names)
            self.assertIn('Documentation_template.md', names)
            self.assertIn('code/business_entity_resolution/src/er/__main__.py', names)
            self.assertIn('code/business_entity_resolution/model/model.txt', names)
            self.assertFalse(any('.venv' in name or '__pycache__' in name for name in names))

    def test_package_rejects_marked_smoke_data(self):
        (self.dataset / 'SMOKE_ONLY.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'smoke'):
            package(self.dataset, self.work, self.output, 'Test', 'Test', self.root / 'fixture.zip')


if __name__ == '__main__':
    unittest.main()

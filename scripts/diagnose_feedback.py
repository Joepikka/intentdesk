"""Reproduce aggregate fit/development diagnostics without printing review text."""
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from intentdesk.common import write_json, sha256_file
from intentdesk.feedback_data import DIRECTORY, load_partition
from intentdesk.feedback_models import classification
from intentdesk.feedback_v2 import checked, features
from intentdesk.feedback_v3 import fit_rows


def main():
    root = Path(__file__).resolve().parents[1]
    directory, metadata = checked(root)
    models = joblib.load(directory/'models.joblib')
    fit = [r for batch in fit_rows(root, limit=10000) for r in batch]
    development = load_partition(root, 'development')
    result = {'partition_use': 'Fit/development only; never opens final-check samples',
        'baseline_run':metadata['run_id'], 'models_sha256':sha256_file(directory/'models.joblib'),
        'development_sha256':sha256_file(root/DIRECTORY/'development.jsonl'),
        'manifest_sha256':sha256_file(root/DIRECTORY/'manifest.json'),
        'fit_sample':'First 10,000 eligible fit hashes in the natural class distribution; in-sample diagnostic',
        'limitations':['Length comparisons are observational, not causal',
            'Subjective labels may be ambiguous; this does not estimate an achievable accuracy ceiling',
            'Training diagnostic is a subset; it is not cross-validation'], 'models':{}}
    for partition, rows in [('fit', fit), ('development', development)]:
        actual = np.array([r['rating'] for r in rows])
        predicted = {n:[] for n in models}
        for start in range(0,len(rows),128):
            with threadpool_limits(limits=2):
                x = features([r['text'] for r in rows[start:start+128]])
                for name, model in models.items():
                    predicted[name].extend(model.predict(x).tolist())
        for name, values in predicted.items():
            p = np.array(values)
            delta = np.abs(actual-p)
            slices = {}
            for label, mask in [('under_200',np.array([len(r['text'])<200 for r in rows])),
                                ('200_to_799',np.array([200<=len(r['text'])<800 for r in rows])),
                                ('800_plus',np.array([len(r['text'])>=800 for r in rows]))]:
                if mask.any():
                    slices[label] = {'count':int(mask.sum()), 'accuracy':float((p[mask]==actual[mask]).mean())}
            result['models'].setdefault(name,{})[partition] = {**classification(actual,p),
                'adjacent_errors':int((delta==1).sum()), 'all_errors':int((delta>0).sum()),
                'mean_absolute_star_error':float(delta.mean()), 'length_slices':slices}
    target = root/'reports/feedback_diagnosis/summary.json'
    write_json(target, result)
    print(f'Aggregate diagnosis saved to {target}; no review text published.')


if __name__ == '__main__':
    main()

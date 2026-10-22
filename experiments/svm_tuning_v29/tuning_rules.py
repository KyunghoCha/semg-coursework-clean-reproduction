"""Small preregistered SVC comparison, retaining the v25 mixture weights."""
CANDIDATES = {
    'regularized': dict(C=10., gamma='scale'),
    'broad': dict(C=100., gamma=1./820.),
    'narrow': dict(C=100., gamma=2./410.),
}
SIZE_CAP = 1.25


def choose(rows):
    reference = next(r for r in rows if r['candidate'] == 'incumbent')
    eligible = [r for r in rows if r['model_bytes'] <= SIZE_CAP * reference['model_bytes']]
    # Prefer the incumbent on equal accuracy. New-candidate ties prefer smaller files,
    # then fewer support vectors and the preregistered name for determinism.
    return min(eligible, key=lambda r: (-r['correct'], r['candidate'] != 'incumbent',
                                      r['model_bytes'], r['support_vectors'], r['candidate']))['candidate']

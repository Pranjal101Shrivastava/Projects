"""Fixture: a suppression with no written reason must NOT be accepted.

The audit tool allows a finding to be acknowledged, but only when the author records why.
A bare `# audit: ok(rule)` with no explanation is exactly the behaviour that makes
suppression mechanisms useless in practice, so the tool rejects it and the finding stands.
"""

from sklearn.preprocessing import StandardScaler

SEED = 0


def build(X):
    # audit: ok(preprocessing-leak)
    scaler = StandardScaler()
    return scaler.fit_transform(X)

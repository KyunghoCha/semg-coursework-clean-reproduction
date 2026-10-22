import numpy as np
from gate_rule import needs_svm,labels


def test_adversarial_boundary_and_tie():
    q=np.zeros((3,8));q[:,:2]=[[.374,.626],[.375,.625],[.375001,.624999]]
    s=np.zeros_like(q);s[:,0]=1
    np.testing.assert_array_equal(needs_svm(q,.2),[False,True,True])
    # At the exact boundary the mixture ties, and argmax changes from class1 to class0.
    np.testing.assert_array_equal(labels(q,.2,lambda m:s[m]),[1,0,0])


def test_random_probabilities_and_skipped_callback():
    rng=np.random.default_rng(42);q=rng.dirichlet(np.full(8,.12),size=4000);s=rng.dirichlet(np.full(8,.12),size=4000)
    for beta in [.05,.1,.2]:
        np.testing.assert_array_equal(labels(q,beta,lambda m:s[m]),((1-beta)*q+beta*s).argmax(1))
    def prohibited(mask):raise AssertionError('No call for beta zero or safe margin')
    np.testing.assert_array_equal(labels(q,0,prohibited),q.argmax(1))
    sure=np.zeros((2,8));sure[:,3]=1;np.testing.assert_array_equal(labels(sure,.2,prohibited),[3,3])

import unittest
import numpy as np
from model_family import compose,training_target,COMPONENTS,diagnostics


class FactorizationTests(unittest.TestCase):
    def test_independent_label_order(self):
        for a in range(2):
            for g in range(4):
                q=compose(np.eye(4)[[g]],np.eye(2)[[a]])
                self.assertEqual(q.argmax(),a*4+g)
        g=np.array([[.1,.2,.4,.3]]);a=np.array([[.6,.4]])
        np.testing.assert_allclose(compose(g,a),np.array([[.06,.12,.24,.18,.04,.08,.16,.12]]),atol=1e-15,rtol=0)

    def test_conditional_soft_joint_without_true_label(self):
        g=np.array([[.45,.35,.1,.1]]);a=np.array([[[.5,.5],[1.,0.],[0.,1.],[1.,0.]]])
        q=compose(g,a)
        np.testing.assert_allclose(q,[[.225,.35,0,.1,.225,0,.1,0]])
        self.assertEqual(q.argmax(),1)  # Soft joint may differ from gesture argmax=0.
        np.testing.assert_allclose(q.reshape(1,2,4).sum(1),g)
        with self.assertRaises(ValueError):compose(g,np.array([[.3,.3]]))

    def test_training_partitions_and_error_axes(self):
        y=np.tile(np.arange(8),3);tr=np.arange(16)
        subsets=[]
        for name in COMPONENTS:
            ids,target,classes=training_target(name,tr,y)
            self.assertEqual(set(np.unique(target)),set(range(classes)))
            self.assertTrue(set(ids)<=set(tr))
            if name.startswith('angle_g'):subsets.extend(ids.tolist())
        self.assertEqual(sorted(subsets),tr.tolist())
        truth=np.array([0,0,0,0]);pred=np.array([0,4,1,5]);q=np.eye(8)[pred]
        d=diagnostics(truth,pred,q)
        self.assertEqual([d[k] for k in ['angle_only_errors','gesture_only_errors','both_wrong']],[1,1,1])


if __name__=='__main__':unittest.main()

"""Logging-only boosting callback; never changes training or uses validation labels."""
import json
import os
from datetime import datetime,timezone
from time import perf_counter
from xgboost.callback import TrainingCallback
from semg.paths import RUN_ROOT


class TrainingProgress(TrainingCallback):
    def __init__(self,name,rounds):
        self.name=name;self.rounds=rounds;self.started=perf_counter()

    def emit(self,completed,state):
        data=dict(progress=self.name,completed_rounds=completed,total_rounds=self.rounds,
                  elapsed_seconds=perf_counter()-self.started,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),state=state)
        print(json.dumps(data),flush=True)
        path=RUN_ROOT/'private/training_progress.json';path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

    def before_training(self,model):
        self.emit(0,'training');return model

    def after_iteration(self,model,epoch,evals_log):
        if (epoch+1)%100==0 or epoch+1==self.rounds:self.emit(epoch+1,'training')
        return False

    def after_training(self,model):
        self.emit(self.rounds,'fit_complete');return model

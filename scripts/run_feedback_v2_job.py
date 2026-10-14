"""Run the one bounded improvement attempt with logs, no AI monitoring loop."""
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import os
from pathlib import Path
import traceback

from intentdesk.common import read_json,write_json
from intentdesk.feedback_lab import training_lock
from intentdesk.feedback_v2 import train,evaluate,current_directory


def main():
    root=Path(__file__).resolve().parents[1]
    local=root/"local/feedback_v2_job";local.mkdir(parents=True,exist_ok=True)
    status={"state":"starting","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
        "bounded_attempt":True,"automatic_actions":False}
    with training_lock(root):
        write_json(local/"status.json",status)
        print(f"Local V2 CPU job started (PID {os.getpid()}); logs: {local/'training.log'}",flush=True)
        with (local/"training.log").open("a",encoding="utf-8",buffering=1) as out, \
                (local/"errors.log").open("a",encoding="utf-8",buffering=1) as errors, \
                redirect_stdout(out),redirect_stderr(errors):
            try:
                status.update(state="training");write_json(local/"status.json",status)
                metadata=train(root)
                status.update(state="evaluating",run_id=metadata["run_id"]);write_json(local/"status.json",status)
                report=evaluate(root)
                marker=read_json(current_directory(root)/"final_experiment.json")
                selected=report["selected_classifier"]
                status.update(state="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                    report=marker["report"],selected_classifier=selected,
                    accuracy=report["classification"][selected]["accuracy"],macro_f1=report["classification"][selected]["macro_f1"])
                write_json(local/"status.json",status)
                print(f"Completed bounded V2 attempt: {selected}; report {marker['report']}",flush=True)
            except BaseException as error:
                status.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error_type=type(error).__name__)
                write_json(local/"status.json",status);traceback.print_exc(file=errors);raise


if __name__=="__main__":
    main()

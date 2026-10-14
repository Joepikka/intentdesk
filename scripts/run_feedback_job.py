"""Run the local continuation without an AI monitoring loop; save UTF-8 logs/status."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import os
from pathlib import Path
import traceback

from intentdesk.common import write_json, read_json
from intentdesk.feedback_lab import start_extension, train, training_lock, directory_for
from intentdesk.feedback_eval import evaluate


def main():
    root=Path(__file__).resolve().parents[1]
    local=root/"local/feedback_job";local.mkdir(parents=True,exist_ok=True)
    status_path=local/"status.json"
    started=datetime.now(timezone.utc).isoformat()
    # Acquire before replacing status/logs so a duplicate cannot hide the active job.
    with training_lock(root):
        status={"state":"starting","pid":os.getpid(),"started_utc":started,
            "minimum_total_epochs":20,"maximum_total_epochs":30,"automatic_actions":False}
        write_json(status_path,status)
        print(f"Local CPU training started (PID {os.getpid()}). Logs: {local/'training.log'}",flush=True)
        with (local/"training.log").open("a",encoding="utf-8",buffering=1) as output, \
                (local/"errors.log").open("a",encoding="utf-8",buffering=1) as errors, \
                redirect_stdout(output),redirect_stderr(errors):
            try:
                print(f"\nJob started {started}",flush=True)
                extension=start_extension(root)
                status.update(state="training",run_id=extension["run_id"])
                write_json(status_path,status)
                train(root)
                status.update(state="evaluating")
                write_json(status_path,status)
                report=evaluate(root)
                marker=read_json(directory_for(root)/"final_experiment.json")
                selected=report["selected_classifier"]
                status.update(state="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                    report=marker["report"],selected_classifier=selected,
                    accuracy=report["classification"][selected]["accuracy"],
                    macro_f1=report["classification"][selected]["macro_f1"])
                write_json(status_path,status)
                print(f"Completed. Actual report: {marker['report']}",flush=True)
            except BaseException as error:
                status.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                    error_type=type(error).__name__,finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(status_path,status)
                traceback.print_exc(file=errors)
                raise


if __name__=="__main__":
    main()

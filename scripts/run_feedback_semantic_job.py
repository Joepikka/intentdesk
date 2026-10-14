"""One resumable CPU semantic/ordinal protocol, with logs and no AI polling."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import traceback

from intentdesk.common import read_json, write_json


@contextmanager
def shared_slot(root,wait,status,status_path,local):
    """Use the existing OS lock while keeping queued jobs free of ML imports."""
    path=root/'local/feedback_training.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size==0:
            handle.write(b'0');handle.flush()
        while True:
            handle.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if not wait:
                    raise ValueError('Another feedback job holds the training lock; use --wait-for-lock to queue.') from None
                if status['state']!='queued':
                    status['state']='queued';write_json(status_path,status)
                    with (local/'training.log').open('a',encoding='utf-8') as stream:
                        stream.write('Queued behind the existing local feedback job; no ML model is loaded while waiting.\n')
                time.sleep(30)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_UN)


def main(wait_for_lock=False):
    root=Path(__file__).resolve().parents[1]
    local=root/"local/feedback_semantic_job"
    local.mkdir(parents=True,exist_ok=True)
    status={"state":"starting","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
            "bounded_attempt":True,"automatic_actions":False}
    status_path=local/'status.json'
    if status_path.exists():
        import psutil
        previous=read_json(status_path)
        if previous.get('state') in {'queued','starting','training','evaluating'} and psutil.pid_exists(previous.get('pid',-1)):
            process=psutil.Process(previous['pid'])
            if any(arg.endswith('run_feedback_semantic_job.py') for arg in process.cmdline()):
                raise ValueError('A semantic job is already queued or running.')
    write_json(status_path,status)
    with shared_slot(root,wait_for_lock,status,status_path,local):
        from intentdesk.feedback_semantic import train,evaluate,current_directory
        with (local/"training.log").open("a",encoding="utf-8",buffering=1) as out, \
                (local/"errors.log").open("a",encoding="utf-8",buffering=1) as errors, \
                redirect_stdout(out),redirect_stderr(errors):
            try:
                status["state"]="training";write_json(local/"status.json",status)
                metadata=train(root)
                status.update(state="evaluating",run_id=metadata["run_id"]);write_json(local/"status.json",status)
                report=evaluate(root)
                marker=read_json(current_directory(root)/"final_experiment.json")
                selected=report["selected_classifier"]
                status.update(state="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                    report=marker["report"],selected_classifier=selected,
                    accuracy=report["classification"][selected]["accuracy"],macro_f1=report["classification"][selected]["macro_f1"])
                write_json(local/"status.json",status)
                print(f"Completed semantic/ordinal experiment: {selected}; report {marker['report']}",flush=True)
            except BaseException as error:
                status.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error_type=type(error).__name__)
                write_json(local/"status.json",status)
                traceback.print_exc(file=errors)
                raise


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-lock',action='store_true')
    main(parser.parse_args().wait_for_lock)

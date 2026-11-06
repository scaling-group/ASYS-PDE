"""Study-local numerical deadline; uses the existing process cleanup helper."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run():
    workspace = Path(os.environ['EVE_WORKSPACE_ROOT'])
    output = Path(os.environ['EVE_EVAL_LOG_ROOT'])
    application = Path(os.environ['EVE_SOLVER_ROOT'])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    error = workspace/'logs/optimize/agent_error.txt'
    receipt = dict(numerical_allowance_seconds=300, model_calls=0, reference_access=False)
    process = None
    try:
        if error.exists():
            raise RuntimeError('model session failed: '+error.read_text().strip())
        from scaling_evolve.providers.agent.drivers._subprocess import kill_process_tree
        # The guard owns native cleanup dependencies; fitting can use the
        # existing numerical module wrapper in its own child interpreter.
        command = [os.environ.get('ASYS_FIT_PYTHON', sys.executable), str(Path(__file__).with_name('evaluate.py')),
                   '--application', str(application), '--output', str(output)]
        process = subprocess.Popen(command, start_new_session=True)
        process.wait(timeout=max(.001, 300-(time.monotonic()-started)))
        if process.returncode:
            raise RuntimeError(f'numerical command exited {process.returncode}')
        if not (output/'score.yaml').is_file() or not (output/'state_dict.pt').is_file():
            raise RuntimeError('missing score or retained state')
        receipt['state'] = 'returned'
    except Exception as error:
        if process is not None and process.poll() is None:
            kill_process_tree(process)
        receipt.update(state='failed', error=f'{type(error).__name__}: {error}')
        score = {key: -1e9 for key in ('score', 'physics', 'ic', 'ic_tangent', 'bc', 'ic_for_selection')}
        score['summary'] = receipt['error']
        (output/'score.yaml').write_text(json.dumps(score, indent=2)+'\n')
        (output/'error.txt').write_text(receipt['error']+'\n')
        (output/'state_dict.pt').unlink(missing_ok=True)
    receipt['elapsed_seconds'] = time.monotonic()-started
    (output/'numerical_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    run()

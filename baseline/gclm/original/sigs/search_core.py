"""Finite released proposal loop with the core-aware fit and judge.

Sampling, clustering, decoder and CMA mechanics come from the released adapter.
Every draw is charged once; no outcome-driven replacement or resume is offered.
"""
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")


def finite_loss(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def fit_eligible(receipt):
    validation = receipt.get("validation") or {}
    return (receipt.get("eligible") is True and receipt.get("failure") is None
            and receipt.get("updates", 0) > 0
            and receipt.get("selected_state") == "last_complete_measured_update"
            and validation.get("resolved") is True
            and finite_loss(validation.get("fine")))


def fit_classification(receipt, planned_updates):
    if not fit_eligible(receipt):
        return "ineligible"
    if receipt["updates"] == planned_updates and receipt.get("stop") == "stage_limits":
        return "completed"
    return "validated_partial"


def choose(rows):
    """Lowest qualified common LOSS, earliest terminal ordinal on exact ties."""
    candidates = []
    for row in rows:
        assessment = row.get("assessment") or {}
        loss = assessment.get("score")
        if (row.get("fit_eligible") is True and row.get("replay_ok") is True
                and assessment.get("status") == "qualified" and finite_loss(loss)):
            candidates.append((loss, row["ordinal"], row["identity"]))
    return min(candidates)[2] if candidates else None


def scientific_assessment(assessment):
    result = copy.deepcopy(assessment)
    result.pop("counters", None)
    result.get("diagnostics", {}).pop("elapsed_seconds", None)
    return result


def replay_matches(first, replay):
    return (first["predictions"] == replay["predictions"]
            and first["beta"] == replay["beta"]
            and scientific_assessment(first["assessment"]) == scientific_assessment(replay["assessment"]))


class Pool:
    """Released top-proxy pool, deduplicated by parameterized structure."""
    def __init__(self, capacity):
        self.capacity, self.entries = capacity, []

    def add(self, row):
        if not row["eligible"]:
            return
        previous = next((r for r in self.entries if r["structure"] == row["structure"]), None)
        if previous and previous["loss"] <= row["loss"]:
            return
        if previous:
            self.entries.remove(previous)
        self.entries.append(row)
        self.entries.sort(key=lambda r:r["loss"])
        del self.entries[self.capacity:]


class Engine:
    def __init__(self, upstream, policy):
        from structure import Decoder, library
        self.policy = policy
        self.spec = policy["proposed_search"]
        self.atoms, self.exclusions = library(upstream)
        self.decoder = Decoder(upstream)
        self.runtime_evidence = dict(release_revision=policy["upstream"]["release_revision"],
            accepted_atoms=len(self.atoms), excluded_atoms=len(self.exclusions),
            checkpoint_loaded=True, grammar_library_loaded=True, upstream_directory=str(upstream),
            sources={n:str(ROOT/n) for n in ("structure.py","expression.py","operators.py","core_fitting.py")})

    def fit(self, root, identity, proposal, phase, deadline, initial=None):
        import torch
        from core_fitting import fit
        from single_anchor import save_profile
        from structure import candidate, domain, formula

        path = root/identity
        path.mkdir(exist_ok=False)
        started = time.monotonic()
        planned = sum(v["updates"] for v in self.policy[phase]["stages"])
        row = dict(identity=identity, eligible=False, fit_eligible=False, phase=phase,
                   proposal=proposal, planned_updates=planned,
                   initialization=initial.name if initial else "fresh_literals",
                   status="started")
        write(path/"proposal.json", proposal)
        write(path/"result.json", row)
        try:
            model = candidate(proposal, beta=1., amplitude=.1)
            if initial is not None:
                if initial.parent.resolve() != root.resolve() or not initial.name.startswith("coarse-"):
                    raise ValueError("Terminal initialization must be from this cold case")
                model.load_state_dict(torch.load(initial/"state_dict.pt", weights_only=True,
                                                map_location="cpu"), strict=True)
            # Do not shorten a fit to squeeze it into the remaining outer budget.
            if deadline-time.monotonic() < self.policy[phase]["seconds"]:
                row.update(status="unvisited", failure="Insufficient full fit reservation")
            else:
                frozen, receipt = fit(model, self.policy, phase)
                certificate = domain(model.expression)
                row.update(fit=receipt, domain=certificate,
                           structure=model.expression.parameterized,
                           status=fit_classification(receipt, planned),
                           completed_fraction=receipt["updates"]/planned)
                if receipt["updates"] > 0:
                    torch.save(model.state_dict(), path/"state_dict.pt")
                    write(path/"formula.json", formula(model))
                admitted = (frozen is not None and fit_eligible(receipt)
                            and certificate["admissible"] is True)
                row.update(eligible=admitted, fit_eligible=admitted,
                           loss=receipt["validation"]["fine"] if admitted else None)
                if not certificate["admissible"]:
                    row.update(status="rejected", failure="Final expression domain unresolved")
                if admitted and phase == "terminal":
                    save_profile(frozen, path/"saved")
        except (ValueError, SyntaxError, RuntimeError, OverflowError, TimeoutError, OSError) as error:
            row.update(status="failed", failure=type(error).__name__+": "+str(error),
                       eligible=False, fit_eligible=False, loss=None)
        row["elapsed_seconds"] = time.monotonic()-started
        write(path/"result.json", row)
        return row

    def public(self, root, row, deadline):
        """One combined 120s cap, unchanged export, two fresh cold children."""
        if not row["fit_eligible"]:
            return dict(assessment=dict(status="unresolved", score=None, reason="Ineligible fit"),
                        replay_ok=False, children=[])
        path = root/row["identity"]
        saved = path/"saved"
        before = {n:(saved/n).read_bytes() for n in ("candidate.json", "state_dict.pt")}
        started = time.monotonic()
        end = min(deadline, started+self.spec["common_assess_and_fresh_replay_seconds_per_terminal"])
        record = dict(assessment=dict(status="unresolved", score=None, reason="Measurement incomplete"),
                      replay_ok=False, children=[])
        results = []
        try:
            for label in ("assessment", "replay"):
                seconds = end-time.monotonic()
                if seconds <= 2:
                    raise TimeoutError("Combined common assessment/replay ceiling")
                target = path/(label+".json")
                child_start = time.monotonic()
                try:
                    proc = subprocess.run([sys.executable, str(ROOT/"evaluate_saved.py"),
                        str(saved), str(target), "--seconds", str(seconds-2)],
                        timeout=seconds, capture_output=True, text=True)
                except subprocess.TimeoutExpired as error:
                    def output(value):
                        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                    (path/(label+".log")).write_text(output(error.stdout)+output(error.stderr))
                    record["children"].append(dict(label=label, exit=None, timed_out=True,
                                                   wall_seconds=time.monotonic()-child_start))
                    raise
                (path/(label+".log")).write_text(proc.stdout+proc.stderr)
                record["children"].append(dict(label=label, exit=proc.returncode,
                                               wall_seconds=time.monotonic()-child_start))
                if proc.returncode:
                    raise RuntimeError("Common measurement child failed: "+label)
                value = json.loads(target.read_text())
                results.append(value)
                if label == "assessment":
                    record["common_observation"] = value["assessment"]
                if value["assessment"]["status"] != "qualified" or not finite_loss(value["assessment"]["score"]):
                    raise ValueError("Common assessment is not qualified: "+label)
            record["replay_ok"] = replay_matches(*results)
            if not record["replay_ok"]:
                raise ValueError("Fresh saved-state replay disagrees")
            record["assessment"] = results[0]["assessment"]
        except (ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as error:
            record["assessment"] = dict(status="unresolved", score=None,
                                          reason=type(error).__name__+": "+str(error))
            record["replay_ok"] = False
        record["saved_bytes_unchanged"] = all((saved/n).read_bytes() == v for n,v in before.items())
        if not record["saved_bytes_unchanged"]:
            record.update(replay_ok=False, assessment=dict(status="rejected", score=None,
                                                          reason="Saved-state bytes changed"))
        if time.monotonic() > end:
            record.update(replay_ok=False, assessment=dict(status="unresolved", score=None,
                                                          reason="Combined measurement ceiling exceeded"))
        record["elapsed_seconds"] = time.monotonic()-started
        write(path/"public.json", record)
        return record


def run_case(root, engine, slots, seed, *, started=None, clock=time.monotonic,
             sampler_factory=None, cma_factory=None):
    """Original broad/focused/CMA mechanics; bounded attempts and full terminal ledger."""
    if sampler_factory is None:
        from structure import Proposals
        sampler_factory = Proposals
    if cma_factory is None:
        import cma
        cma_factory = cma.CMAEvolutionStrategy
    spec = engine.spec
    if slots not in spec["slots"] or seed not in spec["seeds"]:
        raise ValueError("Case outside declared matrix")
    if spec["terminal_initialization"] != "retained coarse raw state from this cold run only":
        raise ValueError("Undeclared terminal initialization")
    root.mkdir(parents=True, exist_ok=False)
    started = clock() if started is None else started
    if not 0 <= started <= clock():
        raise ValueError("Invalid case start")
    end = started+spec["booked_upper_bound_seconds"]
    coarse_end = min(end, started+spec["coarse_seconds"])
    pool = Pool(spec["pool"])
    attempts, terminal, snapshots = [], [], []
    selected_stored = None
    cma_record = dict(started=False)
    failure = None
    health = dict(status="awaiting_actual_coarse_attempt", passed=False,
                  runtime=getattr(engine, "runtime_evidence", {}), actual_attempts_observed=0,
                  contract="Finite eligible actual coarse fit and original proposal/result/raw state/formula persisted; no improvement gate or additional/replacement attempt.")
    write(root/"startup-health.json", health)
    sampler = sampler_factory(engine.atoms, slots, 910000+slots*100+seed, 5)

    def coarse_available():
        return (len(attempts) < spec["attempt_ceiling"]
                and coarse_end-clock() >= engine.policy["coarse"]["seconds"])

    def event(value):
        with (root/"attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(clean(value), allow_nan=False)+"\n")

    def begin(proposal):
        row = dict(identity=f"coarse-{len(attempts):05d}", ordinal=len(attempts),
                   eligible=False, proposal=proposal, status="started", started_seconds=clock()-started)
        attempts.append(row)
        event(dict(event="started", identity=row["identity"], ordinal=row["ordinal"],
                   proposal=proposal, seconds=clock()-started))
        return row

    def finish(row, proposal, decoder_error=None):
        ordinal = row["ordinal"]
        attempt_start = row["started_seconds"]
        if decoder_error is not None:
            path = root/row["identity"]
            path.mkdir(exist_ok=False)
            row.update(status="failed", failure="Decoder: "+str(decoder_error))
            write(path/"proposal.json", proposal)
        else:
            row = engine.fit(root, row["identity"], proposal, "coarse", coarse_end)
        row.update(ordinal=ordinal, started_seconds=attempt_start,
                   attempt_elapsed_seconds=clock()-started-attempt_start)
        write(root/row["identity"]/"result.json", row)
        attempts[ordinal] = row
        health["actual_attempts_observed"] = len(attempts)
        evidence = dict(identity=row["identity"], ordinal=ordinal, status=row["status"],
                        eligible=row["eligible"], proxy=row.get("loss"),
                        persisted_files={n:(root/row["identity"]/n).is_file()
                            for n in ("proposal.json","result.json","state_dict.pt","formula.json")})
        if ordinal == 0:
            health["first_actual_attempt"] = evidence
        if not health["passed"]:
            if row["eligible"] and finite_loss(row.get("loss")) and all(evidence["persisted_files"].values()):
                health.update(status="passed", passed=True, first_valid_coarse_attempt=evidence)
            else:
                health["status"] = "awaiting_valid_actual_coarse_fit"
        write(root/"startup-health.json", health)
        pool.add(row)
        event(dict(event="finished", identity=row["identity"], ordinal=ordinal,
                   status=row["status"], eligible=row["eligible"], loss=row.get("loss"),
                   seconds=clock()-started))
        return row

    try:
        groups = sampler.broad()
        for round_index in range(spec["focused_rounds"]+1):
            count = spec["broad_draws"] if round_index == 0 else spec["focused_draws"]
            if round_index:
                if selected_stored is None or not coarse_available():
                    break
                groups = sampler.refine(selected_stored["proposal"]["memberships"])
            draws = iter(sampler.sample(groups, count))
            for _ in range(count):
                # Check before next(): a drawn proposal is always in the ledger.
                if not coarse_available():
                    break
                proposal = next(draws)
                row = finish(begin(proposal), proposal)
                if row["eligible"] and (selected_stored is None or row["loss"] < selected_stored["loss"]):
                    selected_stored = row
            snapshots.append(dict(stage=f"stored_round_{round_index}", attempted=len(attempts),
                                  incumbent=selected_stored["identity"] if selected_stored else None))
        if selected_stored is not None and coarse_available():
            c = spec["cma"]
            es = cma_factory(selected_stored["proposal"]["latent"], c["sigma"],
                dict(popsize=c["population"], seed=920000+slots*100+seed, verbose=-9, verb_log=0))
            generations = 0
            partial_generation = None
            generated_latents, decoded_attempts = 0, 0
            for generation in range(c["generations"]):
                if not coarse_available():
                    break
                latents, values = es.ask(), []
                generated_latents += len(latents)
                for latent in latents:
                    if not coarse_available():
                        break
                    proposal = dict(kind="decoded", latent=latent, generation=generation)
                    row = begin(proposal)  # Failed decode still consumes an ordinal.
                    decoded_attempts += 1
                    try:
                        proposal["atoms"] = engine.decoder(latent, slots)
                    except (ValueError, RuntimeError) as error:
                        row = finish(row, proposal, decoder_error=error)
                    else:
                        row = finish(row, proposal)
                    values.append(row["loss"] if row["eligible"] else 1e12)
                if len(values) != len(latents):
                    partial_generation = dict(generation=generation, attempted=len(values),
                                              proposed=len(latents), passed_to_tell=False,
                                              unvisited_latents=clean(latents[len(values):]))
                    break
                es.tell(latents, values)
                generations += 1
                if es.stop():
                    break
            cma_record = dict(started=True, generations=generations, stop=clean(dict(es.stop())),
                              initial_stored=selected_stored["identity"], partial_generation=partial_generation,
                              generated_latents=generated_latents, decoded_attempts=decoded_attempts)
    except BaseException as error:
        failure = type(error).__name__+": "+str(error)
        raise
    finally:
        if not health["passed"]:
            health.update(status="unresolved", reason="No valid persisted coarse fit before the fixed loop ended",
                          workflow_failure=failure)
        write(root/"startup-health.json", health)
        snapshots.append(dict(stage="after_latent", attempted=len(attempts),
                              incumbent=pool.entries[0]["identity"] if pool.entries else None))
        write(root/"structural.json", dict(snapshots=snapshots, cma=cma_record,
                                           attempts=len(attempts), pool=pool.entries, failure=failure))
        if failure is not None:
            terminal = [dict(identity=f"terminal-{i:02d}", ordinal=i, status="unvisited",
                reason="Coarse workflow interrupted", fit_eligible=False, replay_ok=False,
                assessment=dict(status="unresolved", score=None)) for i in range(spec["pool"])]
            write(root/"terminal.json", terminal)
            write(root/"result.json", dict(slots=slots, seed=seed, failure=failure,
                  coarse_attempts=len(attempts), public_selected=None, terminal=terminal,
                  elapsed_seconds=clock()-started))

    try:
        for ordinal in range(spec["pool"]):
            identity = f"terminal-{ordinal:02d}"
            retained = pool.entries[ordinal] if ordinal < len(pool.entries) else None
            row = dict(identity=identity, ordinal=ordinal, fit_eligible=False,
                       replay_ok=False, status="unvisited", assessment=dict(status="unresolved", score=None))
            terminal.append(row)
            required = engine.policy["terminal"]["seconds"]+spec["common_assess_and_fresh_replay_seconds_per_terminal"]
            if retained is None:
                row["reason"] = "Coarse pool unfilled"
            elif end-clock() < required:
                row.update(reason="Insufficient full fit and measurement reservation", retained=retained["identity"])
            else:
                row.update(status="started", retained=retained["identity"])
                write(root/"terminal.json", terminal)  # Preserve an interrupted in-flight slot.
                row = engine.fit(root, identity, retained["proposal"], "terminal",
                                 end-spec["common_assess_and_fresh_replay_seconds_per_terminal"],
                                 initial=root/retained["identity"])
                row.update(ordinal=ordinal, retained=retained["identity"])
                terminal[-1] = row
                write(root/"terminal.json", terminal)
                row.update(engine.public(root, row, end))
            write(root/"terminal.json", terminal)
    except BaseException as error:
        failure = type(error).__name__+": "+str(error)
        if terminal:
            terminal[-1].update(status="failed", failure=failure, fit_eligible=False,
                                replay_ok=False, assessment=dict(status="unresolved", score=None))
        raise
    finally:
        while len(terminal) < spec["pool"]:
            ordinal = len(terminal)
            terminal.append(dict(identity=f"terminal-{ordinal:02d}", ordinal=ordinal,
                status="unvisited", reason="Workflow interrupted", fit_eligible=False, replay_ok=False,
                assessment=dict(status="unresolved", score=None)))
        result = dict(mode="single-a core-aware numerical representation comparison", slots=slots, seed=seed,
                      terminal=terminal, public_selected=choose(terminal), coarse_attempts=len(attempts),
                      coarse_eligible=sum(r["eligible"] for r in attempts), elapsed_seconds=clock()-started,
                      cma=cma_record, attempt_ceiling=spec["attempt_ceiling"],
                      maximum_unattempted=spec["attempt_ceiling"]-len(attempts), failure=failure)
        write(root/"terminal.json", terminal)
        write(root/"result.json", result)
    return result

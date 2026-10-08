# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""PhysboOptimizeWorkChain: a closed Bayesian-optimization loop over a PHYSBO test function.

    random phase:  one `propose` (random, num_random points) -> one `evaluate_test_function` -> `observe`
    bayes phase:   num_bayes times: `propose` (score) -> `evaluate_test_function` -> `observe`

Every step is a calcfunction node called by this WorkChain, so the campaign is one provenance graph:
candidates -> proposal -> evaluation -> observations -> next proposal -> ...
The objective is a PHYSBO test function (objectives.py); an external objective (an experiment or a
CalcJob) is driven instead with the `propose` / `observe` CLI tools (interactive mode).
"""
from aiida import orm
from aiida.engine import WorkChain, while_

from .. import objectives
from ..calcfunctions import PROPOSE_DEFAULTS, evaluate_test_function, observe, propose, summarize
from ..data import CandidatesData, ObservationsData


class PhysboOptimizeWorkChain(WorkChain):
    """Bayesian optimization of a PHYSBO test function on a discrete candidate set."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("candidates", valid_type=CandidatesData, help="the search space (rows of X)")
        spec.input("objective", valid_type=orm.Dict,
                   help='{"name": <test function>, "kwargs": {...}, "maximize": false}; see objectives.py')
        spec.input("parameters", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}),
                   help="propose parameters (score, num_rand_basis, num_search_each_probe, seed, ...); "
                        "num_objectives and maximize are set from the objective")
        spec.input("num_random", valid_type=orm.Int, default=lambda: orm.Int(10),
                   help="random evaluations before the Bayesian steps (0 to skip; then `observations` is required)")
        spec.input("num_bayes", valid_type=orm.Int, default=lambda: orm.Int(20), help="number of Bayesian steps")
        spec.input("observations", valid_type=ObservationsData, required=False, help="observations to start from")
        spec.input("label", valid_type=orm.Str, default=lambda: orm.Str("physbo"), help="label of the created nodes")

        spec.outline(cls.setup, cls.run_random, while_(cls.should_continue)(cls.run_bayes_step), cls.finalize)

        spec.output("observations", valid_type=ObservationsData, help="all observations (the final chain node)")
        spec.output("summary", valid_type=orm.Dict, help="best value / Pareto front, best sequence, steps")

        spec.exit_code(400, "ERROR_STEP_FAILED", message="a propose / evaluate / observe step raised")
        spec.exit_code(410, "ERROR_BAD_OBJECTIVE", message="the objective spec is not a known PHYSBO test function")
        spec.exit_code(420, "ERROR_NO_CANDIDATES_LEFT", message="every candidate has been observed")
        spec.exit_code(421, "ERROR_NOTHING_TO_START_FROM", message="num_random is 0 and no observations were given")

    # ------------------------------------------------------------ steps
    def setup(self):
        spec = self.inputs.objective.get_dict()
        try:
            fn = objectives.make(spec)
        except Exception as exc:  # noqa: BLE001
            self.report(f"bad objective {spec!r}: {exc}")
            return self.exit_codes.ERROR_BAD_OBJECTIVE
        if fn.dim != self.inputs.candidates.dim:
            self.report(f"objective dimension {fn.dim} != candidates dimension {self.inputs.candidates.dim}")
            return self.exit_codes.ERROR_BAD_OBJECTIVE
        params = dict(self.inputs.parameters.get_dict())
        unknown = set(params) - set(PROPOSE_DEFAULTS)
        if unknown:
            self.report(f"unknown propose parameters {sorted(unknown)}")
            return self.exit_codes.ERROR_BAD_OBJECTIVE
        params["num_objectives"] = fn.nobj
        params["maximize"] = bool(spec.get("maximize", False))
        params.pop("random", None)
        self.ctx.params = params
        self.ctx.seed = params.get("seed")
        self.ctx.observations = self.inputs.observations if "observations" in self.inputs else None
        self.ctx.step = 0
        self.ctx.num_bayes = self.inputs.num_bayes.value
        if self.inputs.num_random.value <= 0 and self.ctx.observations is None:
            return self.exit_codes.ERROR_NOTHING_TO_START_FROM

    def _propose_params(self, **extra):
        p = dict(self.ctx.params, **extra)
        if self.ctx.seed is not None:
            p["seed"] = int(self.ctx.seed) + self.ctx.step
        return orm.Dict(dict=p)

    def _one_step(self, params: orm.Dict):
        """propose -> evaluate -> observe; returns the summary dict or an exit code."""
        label = self.inputs.label.value
        kwargs = {"candidates": self.inputs.candidates, "parameters": params}
        if self.ctx.observations is not None:
            kwargs["observations"] = self.ctx.observations
        try:
            out = propose(**kwargs, metadata={"label": f"{label}_propose_{self.ctx.step}", "call_link_label": f"propose_{self.ctx.step}"})
            evaluated = evaluate_test_function(self.inputs.candidates, out["proposal"], self.inputs.objective,
                                               metadata={"label": f"{label}_evaluate_{self.ctx.step}",
                                                         "call_link_label": f"evaluate_{self.ctx.step}"})
            okw = {"candidates": self.inputs.candidates, "new": evaluated}
            if self.ctx.observations is not None:
                okw["observations"] = self.ctx.observations
            self.ctx.observations = observe(**okw, metadata={"label": f"{label}_observe_{self.ctx.step}",
                                                             "call_link_label": f"observe_{self.ctx.step}"})
        except ValueError as exc:
            if "observed already" in str(exc):
                self.report(str(exc))
                return self.exit_codes.ERROR_NO_CANDIDATES_LEFT
            self.report(f"step {self.ctx.step} failed: {exc}")
            return self.exit_codes.ERROR_STEP_FAILED
        except Exception as exc:  # noqa: BLE001
            self.report(f"step {self.ctx.step} failed: {type(exc).__name__}: {exc}")
            return self.exit_codes.ERROR_STEP_FAILED
        summary = out["summary"].get_dict()
        self.report(f"step {self.ctx.step} ({summary['mode']}): actions {summary['proposed_actions']}, "
                    f"best so far {summary['best_so_far']}")
        self.ctx.step += 1
        return summary

    def run_random(self):
        n = self.inputs.num_random.value
        if n <= 0:
            return None
        result = self._one_step(self._propose_params(random=True, num_search_each_probe=n))
        if not isinstance(result, dict):
            return result

    def should_continue(self):
        return self.ctx.num_bayes > 0

    def run_bayes_step(self):
        self.ctx.num_bayes -= 1
        result = self._one_step(self._propose_params(random=False))
        if not isinstance(result, dict):
            return result

    def finalize(self):
        obs = self.ctx.observations
        settings = {"objective": self.inputs.objective.get_dict(), "maximize": bool(self.ctx.params["maximize"]),
                    "steps": self.ctx.step, "num_random": self.inputs.num_random.value,
                    "num_bayes": self.inputs.num_bayes.value, "parameters": self.ctx.params}
        summary = summarize(self.inputs.candidates, obs, orm.Dict(dict=settings),
                            metadata={"label": f"{self.inputs.label.value}_summary", "call_link_label": "summarize"})
        self.out("observations", obs)
        self.out("summary", summary)

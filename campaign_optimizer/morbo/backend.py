from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping, Sequence

from .frontier import FrontierRecord, compute_pareto_frontier
from .objectives import (
    ObjectiveSpec,
    TERMINAL_FAILURE_STATUSES,
    TERMINAL_SUCCESS_STATUSES,
)
from .observations import (
    ObservedTrial,
    SkippedTrial,
    TrainingData,
    TrialInput,
    build_observations,
    build_training_data,
    frontier_records_from_observations,
)
from .regions import (
    ACTIVE,
    RESTARTING,
    RegionRecord,
    RegionalPolicy,
    assign_observations_to_regions,
    finalize_region_counts,
    initialize_regions,
    region_bounds,
    restart_regions,
    update_regions,
)
from .botorch_model import (
    BotorchRegionalConfig,
    suggest_botorch_regional_candidates,
)
from .categorical import (
    CategoricalRegionalPolicy,
    apply_categorical_regional_policy,
)
from .search_space import SearchSpaceCodec
from .state import CandidateRegistry, OptimizerState


@dataclass(frozen=True)
class CandidateProposal:
    params: dict[str, Any]
    candidate_signature: str
    region_id: str | None
    strategy: str
    acquisition_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "params": dict(self.params),
            "candidate_signature": self.candidate_signature,
            "region_id": self.region_id,
            "strategy": self.strategy,
        }
        if self.acquisition_value is not None:
            out["acquisition_value"] = float(self.acquisition_value)
        return out


@dataclass(frozen=True)
class BackendSyncResult:
    observations: tuple[ObservedTrial, ...]
    skipped: tuple[SkippedTrial, ...]
    training_data: TrainingData
    frontier: tuple[FrontierRecord, ...]
    regions: tuple[RegionRecord, ...]
    assignments: dict[str, str]
    state: OptimizerState

    @property
    def observed_count(self) -> int:
        return len(self.observations)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


class MorboLikeBackend:
    """MORBO-like regional multi-objective backend without BoTorch.

    This class coordinates the pure components:

    - objective resolution
    - training-data construction
    - Pareto frontier computation
    - trust-region-inspired region refresh
    - seen/pending candidate deduplication
    - global/random and regional/random candidate proposals

    It intentionally does not import or call torch, botorch, SLURM, WarpX,
    HDF5/openPMD readers, or campaign cleanup tools.
    """

    def __init__(
        self,
        space: Mapping[str, Any],
        objective_spec: ObjectiveSpec,
        seed: int = 42,
        min_observations: int | None = None,
        regional_policy: RegionalPolicy | None = None,
        state: OptimizerState | None = None,
        suggestion_mode: str = "regional_random",
        botorch_config: BotorchRegionalConfig | Mapping[str, Any] | None = None,
        categorical_policy: CategoricalRegionalPolicy | Mapping[str, Any] | None = None,
    ):
        self.space_codec = SearchSpaceCodec(space)
        self.objective_spec = objective_spec
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        self.rng = random.Random(self.seed)
        self.min_observations = (
            int(min_observations)
            if min_observations is not None
            else max(
                4,
                len(self.objective_spec.objectives) + 1,
                self.space_codec.encoded_dim() + 1,
            )
        )
        if self.min_observations < 1:
            raise ValueError("min_observations must be >= 1")

        self.regional_policy = regional_policy or RegionalPolicy()
        self.suggestion_mode = str(suggestion_mode or "regional_random")
        if isinstance(botorch_config, BotorchRegionalConfig):
            self.botorch_config = botorch_config
        else:
            self.botorch_config = BotorchRegionalConfig.from_dict(botorch_config)
        if isinstance(categorical_policy, CategoricalRegionalPolicy):
            self.categorical_policy = categorical_policy
        else:
            self.categorical_policy = CategoricalRegionalPolicy.from_dict(
                categorical_policy
            )
        self.last_model_diagnostics: dict[str, Any] = {}
        self.state = state or OptimizerState(
            seed=self.seed,
            objective_spec=self.objective_spec.as_dict(),
            search_space_signature=self.search_space_signature(),
        )

        self.observations: tuple[ObservedTrial, ...] = ()
        self.skipped: tuple[SkippedTrial, ...] = ()
        self.training_data = TrainingData(
            candidate_ids=(),
            params=(),
            encoded_X=(),
            canonical_Y=(),
            objective_names=self.objective_spec.objective_names,
            objective_values=(),
        )
        self.frontier: tuple[FrontierRecord, ...] = ()
        self.regions: tuple[RegionRecord, ...] = tuple(self.state.regions)
        self.assignments: dict[str, str] = {}
        self.last_strategy: str | None = self.state.last_strategy

    def search_space_signature(self) -> str:
        payload = {
            "parameter_names": list(self.space_codec.parameter_names),
            "encoded_dim": self.space_codec.encoded_dim(),
        }
        return repr(payload)

    def sync(
        self,
        trials: Sequence[TrialInput],
        success_statuses: Iterable[str] | None = None,
        proposal_region_map: Mapping[str, str] | None = None,
    ) -> BackendSyncResult:
        """Refresh observations, frontier, regions and optimizer state."""

        success_status_set = set(success_statuses or TERMINAL_SUCCESS_STATUSES)
        proposal_region_map = dict(proposal_region_map or {})

        observation_result = build_observations(
            trials,
            self.objective_spec,
            success_statuses=success_status_set,
        )
        self.observations = observation_result.observations
        self.skipped = observation_result.skipped

        self.training_data = build_training_data(
            self.observations,
            self.space_codec,
            self.objective_spec.objective_names,
        )

        observation_records = frontier_records_from_observations(self.observations)
        self.frontier = tuple(
            compute_pareto_frontier(
                observation_records,
                self.objective_spec.objective_names,
            )
        )

        params_by_candidate_id = {
            observation.candidate_id: observation.params
            for observation in self.observations
        }

        regions = tuple(self.state.regions)
        assignments: dict[str, str] = {}

        if len(self.observations) < self.min_observations:
            regions = ()
        else:
            if not regions:
                regions = tuple(
                    initialize_regions(
                        observation_records,
                        list(self.frontier),
                        params_by_candidate_id,
                        self.space_codec,
                        self.regional_policy,
                    )
                )

            assignments = assign_observations_to_regions(
                observation_records,
                params_by_candidate_id,
                self.space_codec,
                regions,
            )

            regions = tuple(
                update_regions(
                    regions,
                    observation_records,
                    list(self.frontier),
                    assignments,
                    proposal_region_map,
                    params_by_candidate_id,
                    self.space_codec,
                    self.regional_policy,
                )
            )

            if any(region.status == RESTARTING for region in regions):
                regions = tuple(
                    restart_regions(
                        regions,
                        observation_records,
                        list(self.frontier),
                        params_by_candidate_id,
                        self.space_codec,
                        self.regional_policy,
                    )
                )
                assignments = assign_observations_to_regions(
                    observation_records,
                    params_by_candidate_id,
                    self.space_codec,
                    regions,
                )

            regions = tuple(
                finalize_region_counts(
                    regions,
                    observation_records,
                    list(self.frontier),
                    assignments,
                )
            )

        self.regions = tuple(sorted(regions, key=lambda item: item.region_id))
        self.assignments = assignments

        registry = self._sync_candidate_registry(
            trials,
            success_status_set=success_status_set,
        )

        self.state = self.state.with_updates(
            seed=self.seed,
            objective_spec=self.objective_spec.as_dict(),
            search_space_signature=self.search_space_signature(),
            candidate_registry=registry,
            regions=self.regions,
            last_strategy=self.last_strategy,
        )

        return BackendSyncResult(
            observations=self.observations,
            skipped=self.skipped,
            training_data=self.training_data,
            frontier=self.frontier,
            regions=self.regions,
            assignments=dict(self.assignments),
            state=self.state,
        )

    def suggest(self, n: int) -> list[dict[str, Any]]:
        return [proposal.params for proposal in self.suggest_proposals(n)]

    def suggest_proposals(self, n: int) -> list[CandidateProposal]:
        if n <= 0:
            return []

        blocked = self._blocked_signatures()

        if len(self.observations) < self.min_observations:
            proposals = self._sample_unique_random_proposals(
                n,
                strategy="global_random",
                region_id=None,
                blocked_signatures=blocked,
            )
            self.last_strategy = "global_random"
            return proposals

        active_regions = [
            region
            for region in sorted(self.regions, key=lambda item: item.region_id)
            if region.status == ACTIVE
        ]

        if not active_regions:
            proposals = self._sample_unique_random_proposals(
                n,
                strategy="global_random",
                region_id=None,
                blocked_signatures=blocked,
            )
            self.last_strategy = "global_random"
            return proposals

        if self.suggestion_mode in {"regional_model", "botorch", "qlognehvi", "qnehvi"}:
            model_proposals = self._sample_botorch_regional_model_proposals(
                n=n,
                active_regions=active_regions,
                blocked_signatures=blocked,
            )
            if model_proposals:
                batch_signatures = {
                    proposal.candidate_signature for proposal in model_proposals
                }
                proposals = list(model_proposals)

                if len(proposals) < n and self.botorch_config.fallback_to_random:
                    fillers = self._sample_unique_random_proposals(
                        n - len(proposals),
                        strategy="regional_random",
                        region_id=None,
                        blocked_signatures=blocked | batch_signatures,
                    )
                    proposals.extend(fillers)

                self.last_strategy = "regional_model"
                return proposals

            if not self.botorch_config.fallback_to_random:
                self.last_strategy = "regional_model_failed"
                return []

        proposals: list[CandidateProposal] = []
        batch_signatures: set[str] = set()

        for idx in range(n):
            region = active_regions[idx % len(active_regions)]
            proposal = self._sample_one_unique_proposal(
                strategy="regional_random",
                region_id=region.region_id,
                bounds=region_bounds(region),
                blocked_signatures=blocked | batch_signatures,
            )
            if proposal is None:
                continue
            proposals.append(proposal)
            batch_signatures.add(proposal.candidate_signature)

        if len(proposals) < n:
            fillers = self._sample_unique_random_proposals(
                n - len(proposals),
                strategy="global_random",
                region_id=None,
                blocked_signatures=blocked | batch_signatures,
            )
            proposals.extend(fillers)

        self.last_strategy = proposals[0].strategy if proposals else "global_random"
        return proposals

    def register_pending(
        self, proposals: Sequence[CandidateProposal | Mapping[str, Any]]
    ) -> None:
        """Mark proposed candidates as pending and therefore blocked."""

        signatures: list[str] = []
        for proposal in proposals:
            if isinstance(proposal, CandidateProposal):
                signatures.append(proposal.candidate_signature)
            else:
                signatures.append(self.space_codec.signature(proposal))

        registry = self.state.candidate_registry.mark_pending(signatures)
        self.state = self.state.with_updates(candidate_registry=registry)

    def _sample_botorch_regional_model_proposals(
        self,
        n: int,
        active_regions: Sequence[RegionRecord],
        blocked_signatures: set[str],
    ) -> list[CandidateProposal]:
        result = suggest_botorch_regional_candidates(
            encoded_X=self.training_data.encoded_X,
            canonical_Y=self.training_data.canonical_Y,
            objective_names=self.training_data.objective_names,
            codec=self.space_codec,
            active_regions=active_regions,
            rng=self.rng,
            n=n,
            blocked_signatures=blocked_signatures,
            config=self.botorch_config,
            categorical_policy=self.categorical_policy,
        )

        self.last_model_diagnostics = {
            "status": result.status,
            "reason": result.reason,
            **dict(result.diagnostics or {}),
        }

        if result.ref_point_raw or result.ref_point_model_units or result.y_transform:
            self.state = self.state.with_updates(
                ref_point_raw=result.ref_point_raw or {},
                ref_point_model_units=result.ref_point_model_units or {},
                y_transform=result.y_transform or {},
                extra={
                    **dict(self.state.extra),
                    "last_model_diagnostics": self.last_model_diagnostics,
                },
            )

        if result.status != "ok":
            return []

        return [
            CandidateProposal(
                params=candidate.params,
                candidate_signature=candidate.candidate_signature,
                region_id=candidate.region_id,
                strategy="regional_model",
                acquisition_value=candidate.acquisition_value,
            )
            for candidate in result.candidates
        ]

    def _sample_unique_random_proposals(
        self,
        n: int,
        strategy: str,
        region_id: str | None,
        blocked_signatures: set[str],
    ) -> list[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        batch_signatures: set[str] = set()

        while len(proposals) < n:
            proposal = self._sample_one_unique_proposal(
                strategy=strategy,
                region_id=region_id,
                bounds=None,
                blocked_signatures=blocked_signatures | batch_signatures,
            )
            if proposal is None:
                break
            proposals.append(proposal)
            batch_signatures.add(proposal.candidate_signature)

        return proposals

    def _sample_one_unique_proposal(
        self,
        strategy: str,
        region_id: str | None,
        bounds: Sequence[tuple[float, float]] | None,
        blocked_signatures: set[str],
    ) -> CandidateProposal | None:
        max_attempts = 200

        for _attempt in range(max_attempts):
            if bounds is None:
                params = self.space_codec.sample(self.rng)
                params = self.space_codec.project_params(params)
            else:
                params = self.space_codec.sample_within_bounds(bounds, self.rng)
                params = apply_categorical_regional_policy(
                    params,
                    codec=self.space_codec,
                    rng=self.rng,
                    policy=self.categorical_policy,
                )
            signature = self.space_codec.signature(params)
            if signature in blocked_signatures:
                continue

            return CandidateProposal(
                params=params,
                candidate_signature=signature,
                region_id=region_id,
                strategy=strategy,
            )

        return None

    def _blocked_signatures(self) -> set[str]:
        registry = self.state.candidate_registry
        return set(registry.seen_candidate_signatures) | set(
            registry.pending_candidate_signatures
        )

    def _sync_candidate_registry(
        self,
        trials: Sequence[TrialInput],
        success_status_set: set[str],
    ) -> CandidateRegistry:
        registry = self.state.candidate_registry

        all_trial_signatures: list[str] = []
        clear_pending_signatures: list[str] = []

        for trial in trials:
            try:
                signature = self.space_codec.signature(trial.params)
            except Exception:
                continue

            all_trial_signatures.append(signature)

            status = trial.simulation_status
            if status in success_status_set or status in TERMINAL_FAILURE_STATUSES:
                clear_pending_signatures.append(signature)

        registry = registry.mark_seen(all_trial_signatures)
        registry = registry.clear_pending(clear_pending_signatures)
        return registry
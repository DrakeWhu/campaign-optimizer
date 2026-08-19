from __future__ import annotations

import unittest

from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.search_space import FloatRange
from campaign_optimizer.morbo.state import CandidateRegistry


PROPOSED_FOCUS = 0.46876832819742553
OBSERVED_FOCUS = 0.4687683281974255


def _spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score", "metric": "score", "sense": "max"},
            ]
        }
    )


def _backend() -> MorboLikeBackend:
    return MorboLikeBackend(
        {"focus_mm_num": FloatRange(-5.0, 5.0)},
        _spec(),
        seed=17,
        min_observations=2,
    )


class CandidateRegistrySignatureProvenanceTests(unittest.TestCase):
    def test_terminal_trial_clears_original_pending_signature_after_float_roundtrip(
        self,
    ) -> None:
        backend = _backend()
        original_signature = backend.space_codec.signature(
            {"focus_mm_num": PROPOSED_FOCUS}
        )
        recomputed_signature = backend.space_codec.signature(
            {"focus_mm_num": OBSERVED_FOCUS}
        )
        self.assertNotEqual(recomputed_signature, original_signature)

        backend.state = backend.state.with_updates(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(original_signature,),
                pending_candidate_signatures=(original_signature,),
            )
        )
        trial = TrialInput(
            candidate_id="campaign:morbo_001_004",
            params={"focus_mm_num": OBSERVED_FOCUS},
            raw_metrics={"score": 690.829069606},
            simulation_status="finished",
            metadata={"candidate_signature": original_signature},
        )

        result = backend.sync([trial])
        registry = result.state.candidate_registry

        self.assertIn(original_signature, registry.seen_candidate_signatures)
        self.assertNotIn(recomputed_signature, registry.seen_candidate_signatures)
        self.assertNotIn(original_signature, registry.pending_candidate_signatures)

    def test_registry_without_provenance_keeps_historical_recomputation(self) -> None:
        backend = _backend()
        recomputed_signature = backend.space_codec.signature(
            {"focus_mm_num": OBSERVED_FOCUS}
        )
        backend.state = backend.state.with_updates(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(recomputed_signature,),
                pending_candidate_signatures=(recomputed_signature,),
            )
        )
        trial = TrialInput(
            candidate_id="historical:case",
            params={"focus_mm_num": OBSERVED_FOCUS},
            raw_metrics={"score": 690.829069606},
            simulation_status="finished",
        )

        result = backend.sync([trial])
        registry = result.state.candidate_registry

        self.assertIn(recomputed_signature, registry.seen_candidate_signatures)
        self.assertNotIn(recomputed_signature, registry.pending_candidate_signatures)


if __name__ == "__main__":
    unittest.main()

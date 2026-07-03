from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.capillary.batch import recommended_candidates_to_candidate_batch
from campaign_optimizer.morbo.backend import CandidateProposal
from campaign_optimizer.morbo.proposals import (
    DEFAULT_RANKING_SOURCE,
    proposals_to_recommended_dataframe,
    write_recommended_candidates_tsv,
)


CAPILLARY_REQUIRED = (
    "laser_case",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
)


def _proposal(
    *,
    laser_case: str = "f32",
    n0: float = 4.0,
    plateau: float = 10.0,
    diameter: float = 300.0,
    focus: float = 0.0,
    signature: str = "sig_a",
    region_id: str | None = "region_000",
    strategy: str = "regional_random",
) -> CandidateProposal:
    return CandidateProposal(
        params={
            "laser_case": laser_case,
            "n0_1e18cm3": n0,
            "plateau_mm_num": plateau,
            "diameter_um_num": diameter,
            "focus_mm_num": focus,
        },
        candidate_signature=signature,
        region_id=region_id,
        strategy=strategy,
    )


class TestMorboProposalOutput(unittest.TestCase):
    def test_proposals_to_recommended_dataframe_matches_expected_contract(self) -> None:
        df = proposals_to_recommended_dataframe(
            [
                _proposal(
                    signature="sig_a",
                    region_id="region_000",
                    strategy="regional_random",
                ),
                _proposal(
                    laser_case="f20",
                    n0=2.5,
                    plateau=5.0,
                    diameter=180.0,
                    focus=-2.0,
                    signature="sig_b",
                    region_id=None,
                    strategy="global_random",
                ),
            ],
            iteration=3,
            required_parameter_columns=CAPILLARY_REQUIRED,
        )

        self.assertEqual(list(df["rank"]), [1, 2])
        self.assertEqual(
            list(df["candidate_id"]),
            ["morbo_003_000", "morbo_003_001"],
        )
        self.assertEqual(
            list(df["recommendation_id"]),
            ["iter_003_rank_001", "iter_003_rank_002"],
        )
        self.assertEqual(
            list(df["candidate_source"]),
            ["regional_random", "global_random"],
        )
        self.assertEqual(
            list(df["recommendation_backend"]),
            ["morbo_like", "morbo_like"],
        )
        self.assertEqual(
            list(df["surrogate_backend"]),
            ["morbo_like", "morbo_like"],
        )
        self.assertEqual(
            list(df["ranking_source"]),
            [DEFAULT_RANKING_SOURCE, DEFAULT_RANKING_SOURCE],
        )
        self.assertEqual(list(df["acquisition_value"]), [2.0, 1.0])
        self.assertEqual(list(df["candidate_signature"]), ["sig_a", "sig_b"])
        self.assertEqual(list(df["region_id"]), ["region_000", ""])
        self.assertEqual(
            list(df["morbo_strategy"]),
            ["regional_random", "global_random"],
        )

        for column in CAPILLARY_REQUIRED:
            self.assertIn(column, df.columns)

    def test_written_recommended_candidates_tsv_can_feed_capillary_candidate_batch(
        self,
    ) -> None:
        proposals = [
            _proposal(signature="sig_a"),
            _proposal(
                laser_case="f40",
                n0=5.0,
                plateau=20.0,
                diameter=400.0,
                focus=2.0,
                signature="sig_b",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "recommended_candidates.tsv"
            written = write_recommended_candidates_tsv(
                path,
                proposals,
                iteration=1,
                required_parameter_columns=CAPILLARY_REQUIRED,
            )
            recommended = pd.read_csv(written, sep="\t")

        batch = recommended_candidates_to_candidate_batch(
            recommended,
            iteration=1,
            objective_config_id="capillary_objectives_v1",
            parameter_space={
                "version": 1,
                "laser_cases": ["f20", "f32", "f40"],
                "ranges": {
                    "n0_1e18cm3": [0.7, 6.0],
                    "plateau_mm_num": [5.0, 25.0],
                    "diameter_um_num": [150.0, 500.0],
                    "focus_mm_num": [-5.0, 5.0],
                },
            },
            batch_config={
                "case_id_start": 10,
                "plasma_kind": "chan",
                "cap_nr": 192,
                "cap_rmax_um": {"policy": "diameter_factor", "factor": 1.2},
            },
        )

        self.assertEqual(list(batch["CASE_ID"]), [10, 11])
        self.assertEqual(list(batch["LASER_CASE"]), ["f32", "f40"])
        self.assertEqual(
            list(batch["OPT_CANDIDATE_ID"]),
            ["morbo_001_000", "morbo_001_001"],
        )
        self.assertEqual(
            list(batch["OPT_RANKING_SOURCE"]),
            [DEFAULT_RANKING_SOURCE, DEFAULT_RANKING_SOURCE],
        )
        self.assertEqual(float(batch.loc[0, "CAP_RMAX_UM"]), 360.0)

    def test_required_parameter_columns_are_enforced(self) -> None:
        proposal = CandidateProposal(
            params={"laser_case": "f32"},
            candidate_signature="sig_a",
            region_id=None,
            strategy="global_random",
        )

        with self.assertRaisesRegex(ValueError, "missing required parameter"):
            proposals_to_recommended_dataframe(
                [proposal],
                iteration=0,
                required_parameter_columns=CAPILLARY_REQUIRED,
            )

    def test_duplicate_signatures_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate candidate_signature"):
            proposals_to_recommended_dataframe(
                [_proposal(signature="sig_a"), _proposal(signature="sig_a")],
                iteration=0,
            )

    def test_empty_proposals_still_return_contract_columns(self) -> None:
        df = proposals_to_recommended_dataframe(
            [],
            iteration=0,
            required_parameter_columns=CAPILLARY_REQUIRED,
        )

        self.assertTrue(df.empty)
        self.assertIn("recommendation_id", df.columns)
        self.assertIn("laser_case", df.columns)


if __name__ == "__main__":
    unittest.main()

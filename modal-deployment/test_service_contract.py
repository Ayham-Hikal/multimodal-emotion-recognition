import unittest

from service_contract import (
    ModalServiceSettings,
    build_timeline_chunk_response,
    ensure_authorized,
    normalize_source_range,
)


class ModalServiceContractTests(unittest.TestCase):
    def test_requests_without_expected_token_are_allowed(self) -> None:
        settings = ModalServiceSettings(auth_token=None, model_version="meld-v1")

        ensure_authorized(settings, authorization_header=None)

    def test_mismatched_bearer_token_is_rejected(self) -> None:
        settings = ModalServiceSettings(auth_token="top-secret", model_version="meld-v1")

        with self.assertRaises(PermissionError):
            ensure_authorized(settings, authorization_header="Bearer wrong-token")

    def test_source_range_is_normalized_to_a_positive_duration(self) -> None:
        start_ms, end_ms = normalize_source_range(9_000, 3_000)

        self.assertEqual((start_ms, end_ms), (9_000, 9_001))

    def test_chunk_response_adds_aggregate_fields_expected_by_convex(self) -> None:
        response = build_timeline_chunk_response(
            chunk_id="chunk-7",
            model_version="meld-v2",
            inference_result={
                "utterances": [
                    {
                        "start_time": 0.2,
                        "end_time": 1.1,
                        "text": "First sentence.",
                        "emotions": [{"label": "joy", "confidence": 0.91}],
                        "sentiments": [{"label": "positive", "confidence": 0.88}],
                    },
                    {
                        "start_time": 1.3,
                        "end_time": 2.0,
                        "text": "Second sentence.",
                        "emotions": [{"label": "neutral", "confidence": 0.77}],
                        "sentiments": [{"label": "neutral", "confidence": 0.73}],
                    },
                ]
            },
            debug={
                "preprocessingMs": 42,
                "inferenceMs": 1530,
                "framesUsed": 30,
                "audioUsed": True,
                "textUsed": True,
                "videoUsed": True,
            },
        )

        self.assertEqual(response["chunkId"], "chunk-7")
        self.assertEqual(
            response["transcript"], "First sentence. Second sentence."
        )
        self.assertEqual(response["modelVersion"], "meld-v2")
        self.assertEqual(response["debug"]["source"], "modal")
        self.assertEqual(response["debug"]["utteranceCount"], 2)
        self.assertEqual(response["dominantEmotion"], "joy")
        self.assertEqual(response["dominantSentiment"], "positive")


if __name__ == "__main__":
    unittest.main()

import unittest

from service_contract import (
    ModalServiceSettings,
    build_modal_error_payload,
    build_timeline_batch_response,
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

    def test_batch_response_maps_selection_utterances_back_to_chunk_windows(self) -> None:
        response = build_timeline_batch_response(
            model_version="meld-v3",
            selection_source_start_ms=2_000,
            chunks=[
                {
                    "chunkId": "chunk-1",
                    "sourceStartMs": 2_000,
                    "sourceEndMs": 5_000,
                },
                {
                    "chunkId": "chunk-2",
                    "sourceStartMs": 4_500,
                    "sourceEndMs": 8_000,
                },
            ],
            inference_result={
                "utterances": [
                    {
                        "start_time": 0.1,
                        "end_time": 1.1,
                        "text": "Opening line.",
                        "emotions": [{"label": "joy", "confidence": 0.93}],
                        "sentiments": [{"label": "positive", "confidence": 0.87}],
                    },
                    {
                        "start_time": 2.7,
                        "end_time": 3.7,
                        "text": "Boundary line.",
                        "emotions": [{"label": "neutral", "confidence": 0.71}],
                        "sentiments": [{"label": "neutral", "confidence": 0.69}],
                    },
                ]
            },
            debug={
                "downloadMs": 580,
                "ffmpegMs": 140,
                "inferenceMs": 1630,
            },
        )

        self.assertEqual(response["schemaVersion"], 1)
        self.assertEqual(len(response["results"]), 2)
        self.assertEqual(response["results"][0]["chunkId"], "chunk-1")
        self.assertEqual(response["results"][0]["transcript"], "Opening line. Boundary line.")
        self.assertEqual(response["results"][0]["utterances"][1]["start_time"], 2.7)
        self.assertEqual(response["results"][0]["utterances"][1]["end_time"], 3.0)
        self.assertEqual(response["results"][1]["chunkId"], "chunk-2")
        self.assertEqual(response["results"][1]["transcript"], "Boundary line.")
        self.assertEqual(response["results"][1]["utterances"][0]["start_time"], 0.2)
        self.assertEqual(response["results"][1]["utterances"][0]["end_time"], 1.2)

    def test_structured_error_payload_uses_safe_copy(self) -> None:
        payload = build_modal_error_payload(
            code="DOWNLOAD_FAILED",
            detail="HTTP 403 from upstream object store.",
            retryable=True,
        )

        self.assertEqual(payload["code"], "DOWNLOAD_FAILED")
        self.assertTrue(payload["retryable"])
        self.assertEqual(
            payload["message"], "We couldn't access the source media for analysis."
        )
        self.assertEqual(payload["detail"], "HTTP 403 from upstream object store.")
        self.assertEqual(payload["schemaVersion"], 1)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import server


class SourceBundleTests(unittest.TestCase):
    def test_quality_score_prefers_useful_korean_transcription(self):
        empty = {'segments': []}
        lyrics = {
            'segments': [
                {'text': '오늘은 바람이 좋아', 'start': 0, 'end': 2},
                {'text': '우리 함께 노래해', 'start': 2, 'end': 4},
            ]
        }

        self.assertGreater(server._transcription_quality(lyrics), server._transcription_quality(empty))

    def test_bundle_selects_candidate_with_higher_transcription_quality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / 'sources.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('weak.wav', b'audio')
                archive.writestr('strong.wav', b'audio')

            def transcribe(path, _extension):
                if path.name == 'weak.wav':
                    return {'segments': [{'text': '음', 'start': 0, 'end': 1}], 'sourceMethod': 'whisper-dtw'}
                return {
                    'segments': [
                        {'text': '오늘은 바람이 좋아', 'start': 0, 'end': 2},
                        {'text': '우리 함께 노래해', 'start': 2, 'end': 4},
                    ],
                    'sourceMethod': 'whisper-dtw',
                }

            with patch.object(server, 'transcribe_media', side_effect=transcribe), patch.object(
                server, '_retain_source_file', return_value='a' * 32
            ):
                result = server.transcribe_bundle(archive_path)

        self.assertEqual(result['sourceFile']['name'], 'strong.wav')
        self.assertGreater(result['sourceFile']['qualityScore'], 0)

    def test_bundle_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / 'unsafe.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('../outside.wav', b'not extracted')

            with self.assertRaises(server.ApiError):
                server.transcribe_bundle(archive_path)


if __name__ == '__main__':
    unittest.main()
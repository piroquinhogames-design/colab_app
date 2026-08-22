import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('STUDIO_PASSWORD', 'search-test-password')
os.environ.setdefault('STUDIO_ROOT', '/tmp/modellab-search-tests')
os.environ.setdefault('MEGA_EMAIL', '')
os.environ.setdefault('MEGA_PASSWORD', '')

sys.path.insert(0, os.path.dirname(__file__))
import server  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def image(image_id, base_model='Anima', created_at='2026-08-22T20:00:00.000Z', prompt='cinematic portrait', reactions=0, resources=None, meta=True):
    return {
        'id': image_id,
        'url': f'https://image.civitai.com/{image_id}.jpg',
        'width': 1024,
        'height': 1024,
        'createdAt': created_at,
        'username': 'artist',
        'baseModel': base_model,
        'nsfw': False,
        'stats': {'heartCount': reactions},
        'meta': ({
            'prompt': prompt,
            'negativePrompt': 'low quality',
            'civitaiResources': resources or [{'type': 'checkpoint', 'modelVersionId': 100 + image_id}],
        } if meta else None),
    }


class SearchEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = server.app.test_client()
        response = cls.client.post('/api/login', json={'password': 'search-test-password'})
        if response.status_code != 200:
            raise AssertionError(response.get_json())

    def test_anima_compatibility_uses_top_level_base_model(self):
        payload = {'items': [image(1)], 'metadata': {'nextCursor': None}}
        with patch.object(server.requests, 'get', return_value=FakeResponse(payload)) as mocked:
            response = self.client.get('/api/prompt-store?family=anima&base_filter=compatible&limit=24')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['id'] for item in response.get_json()['items']], [1])
        self.assertEqual(mocked.call_args.kwargs['params']['baseModels'], 'Anima')

    def test_incompatible_top_level_base_model_is_removed(self):
        payload = {'items': [image(2, base_model='Pony')], 'metadata': {'nextCursor': None}}
        with patch.object(server.requests, 'get', return_value=FakeResponse(payload)):
            response = self.client.get('/api/prompt-store?family=anima&base_filter=compatible')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['items'], [])

    def test_search_uses_resource_names_and_handles_missing_meta(self):
        payload = {'items': [
            image(3, resources=[{
                'type': 'checkpoint', 'modelVersionId': 103,
            }, {
                'type': 'lora', 'modelVersionId': 203, 'modelId': 303,
                'modelName': 'Cinematic Style', 'weight': 2.0,
            }]),
            image(4, meta=False),
        ], 'metadata': {'nextCursor': None}}
        with patch.object(server.requests, 'get', return_value=FakeResponse(payload)):
            response = self.client.get('/api/prompt-store?family=anima&query=cinematic&base_filter=compatible')
        self.assertEqual(response.status_code, 200)
        items = response.get_json()['items']
        self.assertEqual([item['id'] for item in items], [3])
        self.assertEqual(items[0]['loras'][0]['weight'], 1.5)

    def test_oldest_sort_is_sent_to_api_and_applied_locally(self):
        payload = {'items': [
            image(5, created_at='2026-08-22T20:00:00.000Z', reactions=1),
            image(6, created_at='2025-01-01T20:00:00.000Z', reactions=2),
        ], 'metadata': {'nextCursor': None}}
        with patch.object(server.requests, 'get', return_value=FakeResponse(payload)) as mocked:
            response = self.client.get('/api/prompt-store?family=anima&sort=Oldest&base_filter=compatible&limit=2')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['id'] for item in response.get_json()['items']], [6, 5])
        self.assertEqual(mocked.call_args.kwargs['params']['sort'], 'Oldest')

    def test_local_filter_paginates_until_page_is_filled_and_deduplicates(self):
        first_page = {
            'items': [image(7, base_model='Pony'), image(8)],
            'metadata': {'nextCursor': 'page-2'},
        }
        second_page = {
            'items': [image(8), image(9, created_at='2025-01-01T20:00:00.000Z')],
            'metadata': {'nextCursor': None},
        }

        def fake_get(_url, params=None, **_kwargs):
            if params.get('cursor') == 'page-2':
                return FakeResponse(second_page)
            return FakeResponse(first_page)

        with patch.object(server.requests, 'get', side_effect=fake_get) as mocked:
            response = self.client.get('/api/prompt-store?family=anima&base_filter=compatible&limit=2')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['id'] for item in response.get_json()['items']], [8, 9])
        self.assertEqual(mocked.call_count, 2)

    def test_invalid_date_is_rejected_before_external_request(self):
        with patch.object(server.requests, 'get') as mocked:
            response = self.client.get('/api/prompt-store?date_from=2026-99-99')
        self.assertEqual(response.status_code, 400)
        mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()

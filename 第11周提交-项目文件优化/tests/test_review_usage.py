import unittest
from types import SimpleNamespace
from review_usage import UsageRecorder

class UsageTests(unittest.TestCase):
    def test_records_provider_usage_without_guessing_price(self):
        recorder=UsageRecorder()
        wrapped=recorder.wrap(lambda:SimpleNamespace(usage={'prompt_tokens':12,'completion_tokens':3}))
        wrapped()
        result=recorder.summary()
        self.assertEqual(result['requests'],1)
        self.assertEqual(result['provider_usage'][0]['prompt_tokens'],12)
        self.assertIsNone(result['cost'])

    def test_failure_is_counted_without_fabricated_usage(self):
        recorder=UsageRecorder()
        def fail(): raise RuntimeError('failed')
        with self.assertRaises(RuntimeError):recorder.wrap(fail)()
        self.assertEqual(recorder.summary()['requests'],1)
        self.assertEqual(recorder.summary()['status'],'unavailable')

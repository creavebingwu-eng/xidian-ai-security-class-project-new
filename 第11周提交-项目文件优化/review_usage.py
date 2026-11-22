"""Capture provider usage for the standalone reviewer, without guessing prices."""
import time

class UsageRecorder:
    def __init__(self):
        self.started = time.monotonic()
        self.requests = 0
        self.responses = []

    def wrap(self, call):
        def recorded(*args, **kwargs):
            self.requests += 1
            response = call(*args, **kwargs)
            usage = getattr(response,'usage',None)
            if usage is not None:
                self.responses.append(usage.model_dump() if hasattr(usage,'model_dump') else dict(usage))
            return response
        return recorded

    def summary(self):
        return {'status':'recorded' if self.responses else 'unavailable',
                'requests':self.requests,'provider_usage':self.responses,
                'elapsed_seconds':round(time.monotonic()-self.started,3),
                'cost':None,'cost_status':'not_calculated; use provider billing',
                'scope':'standalone make_llm_call; other SDK paths may be unmetered'}

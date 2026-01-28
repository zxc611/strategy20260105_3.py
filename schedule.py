# local stub for schedule to satisfy import in offline environment
class SchedulerStub:
    def __getattr__(self, name):
        raise ImportError("schedule not installed")

# expose commonly used callables as no-ops
run_pending = SchedulerStub()
every = SchedulerStub()

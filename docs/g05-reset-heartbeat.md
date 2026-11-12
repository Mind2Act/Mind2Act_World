# G0.5 connection timing

G0.5 adapter construction now stores configuration only. The first RPC (normally
`reset`) opens the socket and validates the execution-chunk handshake, after
`run_episode` finishes task build, robot initialization and simulator readiness.
A build failure therefore does not open a model connection.

The E1 hard failures inspected in the `g05-clean60e` archives all occurred at
logical tick zero, during the first reset. Of 47 retained records inspected,
27 explicitly reported server close code 1011 / keepalive ping timeout; their
scene builds took 55.0–74.0 seconds. The other 20 reported an unframed disconnect.
Those 20 are not independently attributed to keepalive.

The CPU fake service runs in another process with a 50 ms ping interval and
100 ms pong budget. A 600 ms `ctypes.PyDLL` call holds the environment GIL during
`task.build`, starving the sync WebSocket receiver thread; a normal `time.sleep`
releases the GIL. Before the change, only the native-block case fails at reset
with the same 1011 error. After the change, both complete a synthetic runner
with one connection, a confirmed reset and four physics ticks.

This addresses the connection's exposure to blocking initialization. It does
not change server settings, model identity, action/cache semantics, request
budgets or reconnect a failed trial. Mid-episode GIL starvation remains subject
to the existing server heartbeat budget; no GPU profiling or replay was run.

CPU regression: `python -m pytest tests/test_g05_heartbeat.py tests/test_xpolicylab.py tests/test_xpolicy_failures.py tests/test_xpolicy_seed_identity.py tests/test_xpolicy_result.py -q`.

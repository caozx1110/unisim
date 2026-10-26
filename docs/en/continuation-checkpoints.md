# Continuation checkpoints

[中文](../zh/continuation-checkpoints.md)

`SimBackend.export_continuation_state()` returns detached CPU arrays at a completed step boundary. `import_continuation_state(state)` restores compatible state into an initialized backend, preserving native array addresses and tensor aliases. Other adapters raise `NotImplementedError`; playback snapshots are insufficient. The current MJWarp implementation also rejects fixed model variants and custom host/native physics callbacks, whose extra state has no checkpoint contract yet.

The MJWarp source implementation binds the schema to the compiled model, native package versions, world/capacity settings, graph mode and substep-view bindings. It saves all native Data arrays (including solver warmstart, sleep, contact and constraint state), randomized Model fields and derived constants, adapter caches, pending forces, and substep acceleration/contact history. Import validates all fields before writing, then copies in place without reset or forward. A native copy failure faults the backend. Read-only constructor arrays must match exactly. CUDA graphs, events and native handles are reconstructed rather than serialized.

The owner must stop stepping, persist task/manager state, replay, learner and every RNG separately, and publish a complete checkpoint atomically. No engine RNG is introduced by this adapter. CPU memory is one detached snapshot plus at most one temporary native field; `nbytes` and `nbytes_by_field` report the payload size. State from another model, native version or graph mode is rejected.

This optional source API has CPU validation coverage. A real new-process CUDA continuation comparison remains required for each supported training owner before claiming exact recovery. Rebuilding graphs does not establish numerical trajectory equivalence by itself; tests must compare the saved state and subsequent transitions, including contacts, partial resets, randomized properties and substep histories.

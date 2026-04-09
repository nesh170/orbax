# Single Replica Load Handler: Dispatcher vs Non-Dispatcher Analysis

## Overview

`SingleReplicaArrayHandler` optimizes checkpoint reads in multi-host/multi-pod training by reading data on **one replica only** and broadcasting to the rest, avoiding redundant reads across all hosts. The dispatcher controls *how* that read and broadcast happens.

**Relevant files:**
- Handler: `checkpoint/orbax/checkpoint/_src/serialization/jax_array_handlers.py`
- Dispatcher: `checkpoint/orbax/checkpoint/_src/multihost/dispatchers.py`
- Broadcast utils: `checkpoint/orbax/checkpoint/_src/multihost/multislice.py`

---

## Path 1: Non-Dispatcher (`_single_replica_deserialize_and_broadcast`)

### Host participation
Every host runs code in this path.

```
Primary replica hosts:   _deserialize_arrays()  →  real data from storage
Non-primary hosts:       create_zeros()          →  jnp.zeros (same shape/dtype)
```

Non-primary hosts don't read storage but still allocate zero arrays on-device (explicitly to avoid host RAM spikes).

### Broadcast mechanism

`broadcast_one_replica_to_all` is a **memory-batched collective** that chunks through arrays:

```
1. Compute memory_limit_bytes (fraction of available accelerator memory)
2. For each chunk that fits in memory:
     _globalize_single_replica_arrays()  →  expand replica axis
     _merge_globalized_replicas()        →  jnp.sum(axis=0) via jit
     delete intermediates immediately    →  free memory
     block_until_ready()                 →  sync before next batch
```

**`_globalize_single_replica_arrays`** prepends the replica axis:
```
local shape [D]  →  global shape [num_replicas, D]
```
Primary hosts inject real data into their shards. Non-primary hosts fill with on-device zeros. `jax.make_array_from_single_device_arrays` assembles the global array.

**`_merge_globalized_replicas`** uses a reduce trick:
```python
jnp.sum(axis=0)  # sum over replica axis
# real data + zeros from non-primary replicas = real data on every replica
```
This is a sum-reduce used as a proxy for broadcast — no true all-gather collective.

---

## Path 2: ColocatedPythonDispatcher

### Host participation
Only primary replica hosts run the worker function. Non-primary hosts are idle until the final `jax.device_put`.

### Execution flow

```
1. primary_replica_devices = replica_devices(mesh, replica_id=0)
2. dummy_input_array = get_dummy_input_array(primary_replica_devices)
   → small bool array scoped to primary replica only
   → tells colocated_python which workers to activate

3. dispatcher.dispatch():
   a. _transform_pytree_shardings(func_kwargs)
      → convert every Sharding/RestoreArgs/jax.Array to CPU equivalent
      → uses cp.colocated_cpu_devices(sharding.mesh)

   b. @cp.colocated_python wraps the worker function
      → schedules Python on CPU workers colocated with primary replica accelerators

   c. .specialize(out_specs_fn=lambda _: cpu_result_specs)
      → JAX needs output shape/sharding declared before execution

   d. to_colocated_python(input_arrays)
      → jax.device_put(dummy_array, cpu_sharding)
      → triggers colocated_python execution

   e. _single_replica_deserialize_on_worker runs
      → asyncio_utils.run_sync(_deserialize_arrays(...))
      → CPU-side, reads from storage into CPU memory

   f. _to_final_specs(result, result_specs)
      → jax.device_put(leaf, single_replica_sharding)
      → moves CPU arrays to TPU/GPU on primary replica

4. jax.tree.map(jax.device_put, ret, shardings)
   → ret has single_replica_sharding
   → shardings is the full multi-replica sharding
   → JAX native resharding broadcasts to all devices

5. jax.block_until_ready(ret)
```

---

## Side-by-Side Comparison

| Dimension | Non-Dispatcher | ColocatedPythonDispatcher |
|---|---|---|
| **Non-primary host work** | Creates zero arrays, participates in collective | Idle until final `device_put` |
| **Broadcast mechanism** | Explicit: globalize → sum(axis=0) reduce trick | JAX-native: `device_put` reshard with full sharding |
| **Memory batching** | Yes — chunks by `memory_limit_bytes`, deletes intermediates | No — all arrays at once |
| **Memory safety** | High — explicit OOM guard, per-batch `block_until_ready` | Lower — depends on JAX resharding |
| **Data path** | Storage → host RAM → accelerator → collective | Storage → colocated CPU → primary TPU → reshard |
| **Pathways compatible** | No | Yes — required |
| **Setup overhead** | None | Sharding conversion for every arg + `@cp.colocated_python` specialization |
| **Broadcast control** | Full — `memory_limit_bytes`, `memory_scaling_factor` knobs | None — JAX handles it |
| **Intermediate allocations** | Zeros on non-primary device shards (then freed) | None on non-primary hosts |
| **Collective type** | Custom: make_array + jnp.sum trick | Standard JAX reshard (may use ICI/DCN all-gather internally) |

---

## When to Use Each

| Scenario | Use |
|---|---|
| Running on Pathways runtime | Dispatcher (required — no choice) |
| Very large models, memory constrained | Non-dispatcher (batching is safer) |
| Standard JAX, single or few pods | Non-dispatcher (simpler, lower overhead) |
| Large scale, Pathways-adjacent infra | Dispatcher |

---

## What is Pathways?

Pathways is Google's large-scale distributed ML runtime that treats an entire fleet of TPUs across multiple pods as a **single logical computer**.

**Key differences from standard JAX multi-host:**

| | Standard JAX | Pathways |
|---|---|---|
| Python processes | Each host runs independently | Single client controls many pods |
| Worker autonomy | Hosts run free Python | Workers managed by Pathways runtime |
| Host-side I/O | Free Python execution | Must go through `colocated_python` |

In standard JAX, each host freely decides whether it's a primary replica and runs its own Python code accordingly. **Pathways workers can't do this** — they are not independently running Python programs.

`colocated_python` is the sanctioned mechanism for running Python-side work (like checkpoint I/O) on Pathways workers. The `ColocatedPythonDispatcher` wraps this mechanism to make checkpointing work in that environment.

---

## Core Architectural Difference

**Non-dispatcher** treats broadcast as an explicit **application-level collective** — manually orchestrates which hosts contribute real data vs zeros, then sums. Gives precise memory control at the cost of all hosts doing work.

**Dispatcher** treats broadcast as a **JAX resharding problem** — deserialize only on primary replica's CPU workers, then let JAX's native device placement propagate data. Cleaner separation (only primary hosts do I/O) but loses explicit memory batching and requires the Pathways-compatible `colocated_python` mechanism.

**The missing memory batching in the dispatcher path is the most significant gap** — for very large model checkpoints, the non-dispatcher's chunked broadcast loop with explicit `block_until_ready` per batch is meaningfully safer against OOM.

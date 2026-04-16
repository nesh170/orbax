# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark comparing JAX async overlapping D2H vs torch DCP-style sequential D2H.

Demonstrates that Orbax's async overlapping approach is significantly faster
than PyTorch DCP's sequential blocking approach for device-to-host transfers.

Key difference:
- JAX async (Orbax): Kicks off all D2H transfers concurrently, overlapping
  them. Total time ≈ time of the single slowest transfer.
- Torch DCP-style: Transfers each array sequentially, blocking until complete
  before starting the next. Total time ≈ sum of all individual transfer times.
"""

from collections.abc import Sequence
import dataclasses

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


JAX_ASYNC = 'jax_async'
TORCH_DCP_STYLE = 'torch_dcp_style'


def jax_async_d2h(
    arrays: list[jax.Array],
    enable_pinned_host_transfer: bool = False,
) -> list[np.ndarray]:
  """Orbax async overlapping D2H.

  Kicks off all device-to-host transfers concurrently, then waits for all to
  complete. Total time ≈ time of the single slowest transfer, since the
  individual transfers overlap.

  Args:
    arrays: The JAX arrays to transfer to host.
    enable_pinned_host_transfer: Whether to use pinned host memory for the
      transfer (when available). Pinned memory can improve transfer throughput
      on accelerators.

  Returns:
    The transferred arrays as numpy ndarrays.
  """
  started = []
  for arr in arrays:
    if enable_pinned_host_transfer:
      devices = list(arr.devices())
      device = devices[0] if devices else None
      if device is not None and hasattr(device, 'addressable_memories'):
        has_pinned = any(
            m.kind == 'pinned_host' for m in device.addressable_memories()
        )
        if has_pinned:
          arr = jax.device_put(
              arr,
              jax.sharding.SingleDeviceSharding(
                  device, memory_kind='pinned_host'
              ),
          )
          started.append(arr)
          continue
    # Start the async D2H transfer without blocking.
    arr.copy_to_host_async()
    started.append(arr)
  # Wait for all transfers to complete. Because all transfers were initiated
  # above, they can proceed in parallel on the device side.
  return [np.asarray(a) for a in started]


def torch_dcp_style_d2h(arrays: list[jax.Array]) -> list[np.ndarray]:
  """Torch DCP-style sequential blocking D2H.

  Simulates PyTorch DCP's approach: each array is transferred one at a time,
  blocking until the transfer completes before moving to the next array.
  Total time ≈ sum of all individual transfer times.

  Args:
    arrays: The JAX arrays to transfer to host.

  Returns:
    The transferred arrays as numpy ndarrays.
  """
  results = []
  for arr in arrays:
    # No async prefetch — block until this array is fully transferred before
    # starting the next, mirroring PyTorch DCP's tensor.cpu() behavior.
    results.append(np.asarray(arr))
  return results


@dataclasses.dataclass(frozen=True)
class D2HBenchmarkOptions(core.BenchmarkOptions):
  """Options for D2H comparison benchmarks.

  Each attribute can be a single value or a list of values to create a
  parameter sweep.

  Attributes:
    approach: The D2H transfer approach to benchmark. Use 'jax_async' for
      Orbax's async overlapping approach, or 'torch_dcp_style' for sequential
      blocking transfers (simulating PyTorch DCP's tensor.cpu() pattern).
    enable_pinned_host_transfer: Whether to use pinned host memory for the
      JAX async transfer path. Pinned memory can significantly improve D2H
      throughput on accelerators. Only valid when approach='jax_async'.
  """

  approach: str | Sequence[str] = JAX_ASYNC
  enable_pinned_host_transfer: bool | Sequence[bool] = False

  def is_valid(self) -> bool:
    assert isinstance(self.approach, str)
    assert isinstance(self.enable_pinned_host_transfer, bool)
    if self.approach not in (JAX_ASYNC, TORCH_DCP_STYLE):
      return False
    # Pinned host transfer is only applicable to the JAX async approach.
    if self.enable_pinned_host_transfer and self.approach != JAX_ASYNC:
      return False
    return True


@core.benchmark_options(D2HBenchmarkOptions)
class D2HBenchmark(core.BenchmarksGenerator):
  """Benchmarks D2H: JAX async overlapping (Orbax) vs torch DCP-style sequential.

  Compares Orbax's async overlapping D2H with the sequential blocking approach
  used by PyTorch DCP, demonstrating the performance advantage of overlapping
  device-to-host transfers across many arrays.

  Use with a CheckpointConfig spec containing multiple large arrays to see the
  most pronounced difference between the two approaches.
  """

  def test_fn(self, context: core.TestContext) -> core.TestResult:
    """Runs the D2H benchmark for a single configuration.

    Args:
      context: Test context with a pytree of JAX arrays and benchmark options.

    Returns:
      TestResult with a 'd2h_time_duration' metric capturing the total wall
      time for the device-to-host transfer of all arrays in the pytree.
    """
    metrics = metric_lib.Metrics()
    options = context.options
    assert isinstance(options, D2HBenchmarkOptions)

    arrays = [
        v
        for v in jax.tree_util.tree_leaves(context.pytree)
        if isinstance(v, jax.Array)
    ]

    if not arrays:
      logging.warning('No JAX arrays found in pytree; skipping D2H benchmark.')
      return core.TestResult(metrics=metrics)

    total_bytes = sum(a.nbytes for a in arrays)
    logging.info(
        'D2H benchmark: approach=%s, enable_pinned_host_transfer=%s,'
        ' num_arrays=%d, total_bytes=%d',
        options.approach,
        options.enable_pinned_host_transfer,
        len(arrays),
        total_bytes,
    )

    if options.approach == JAX_ASYNC:
      with metrics.measure('d2h'):
        jax_async_d2h(arrays, options.enable_pinned_host_transfer)
    else:
      with metrics.measure('d2h'):
        torch_dcp_style_d2h(arrays)

    return core.TestResult(metrics=metrics)

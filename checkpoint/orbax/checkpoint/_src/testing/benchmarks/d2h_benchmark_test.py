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

"""Tests for d2h_benchmark."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint._src.testing.benchmarks import d2h_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


D2HBenchmarkOptions = d2h_benchmark.D2HBenchmarkOptions
D2HBenchmark = d2h_benchmark.D2HBenchmark
JAX_ASYNC = d2h_benchmark.JAX_ASYNC
TORCH_DCP_STYLE = d2h_benchmark.TORCH_DCP_STYLE


class D2HBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.pytree = {
        'param_0': jnp.ones((64, 64)),
        'param_1': jnp.zeros((128, 128)),
    }

  def _make_context(
      self, options: D2HBenchmarkOptions
  ) -> benchmarks_core.TestContext:
    return benchmarks_core.TestContext(
        pytree=self.pytree,
        path=self.directory,
        options=options,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='jax_async',
          options=D2HBenchmarkOptions(approach=JAX_ASYNC),
      ),
      dict(
          testcase_name='torch_dcp_style',
          options=D2HBenchmarkOptions(approach=TORCH_DCP_STYLE),
      ),
      dict(
          testcase_name='jax_async_with_pinned',
          options=D2HBenchmarkOptions(
              approach=JAX_ASYNC, enable_pinned_host_transfer=True
          ),
      ),
  )
  def test_benchmark_returns_d2h_time_metric(self, options):
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    context = self._make_context(options)

    result = benchmark.test_fn(context)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertIn('d2h_time_duration', result.metrics.results)

  def test_jax_async_approach_calls_jax_async_d2h(self):
    options = D2HBenchmarkOptions(approach=JAX_ASYNC)
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    with mock.patch.object(
        d2h_benchmark, 'jax_async_d2h', wraps=d2h_benchmark.jax_async_d2h
    ) as mock_jax_d2h:
      context = self._make_context(options)
      result = benchmark.test_fn(context)

    mock_jax_d2h.assert_called_once()
    self.assertIn('d2h_time_duration', result.metrics.results)

  def test_torch_dcp_style_approach_calls_torch_dcp_style_d2h(self):
    options = D2HBenchmarkOptions(approach=TORCH_DCP_STYLE)
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    with mock.patch.object(
        d2h_benchmark,
        'torch_dcp_style_d2h',
        wraps=d2h_benchmark.torch_dcp_style_d2h,
    ) as mock_torch_d2h:
      context = self._make_context(options)
      result = benchmark.test_fn(context)

    mock_torch_d2h.assert_called_once()
    self.assertIn('d2h_time_duration', result.metrics.results)

  def test_jax_async_passes_pinned_flag(self):
    options = D2HBenchmarkOptions(
        approach=JAX_ASYNC, enable_pinned_host_transfer=True
    )
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    with mock.patch.object(
        d2h_benchmark, 'jax_async_d2h', wraps=d2h_benchmark.jax_async_d2h
    ) as mock_jax_d2h:
      context = self._make_context(options)
      benchmark.test_fn(context)

    call_args = mock_jax_d2h.call_args
    # enable_pinned_host_transfer is the second positional argument.
    self.assertTrue(call_args[0][1])

  def test_empty_pytree_skips_d2h_and_returns_no_metric(self):
    options = D2HBenchmarkOptions(approach=JAX_ASYNC)
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    context = benchmarks_core.TestContext(
        pytree={},
        path=self.directory,
        options=options,
    )

    result = benchmark.test_fn(context)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertNotIn('d2h_time_duration', result.metrics.results)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_approach',
          options=D2HBenchmarkOptions(approach=JAX_ASYNC),
          expected_len=1,
      ),
      dict(
          testcase_name='both_approaches',
          options=D2HBenchmarkOptions(approach=[JAX_ASYNC, TORCH_DCP_STYLE]),
          expected_len=2,
      ),
      dict(
          testcase_name='jax_async_with_and_without_pinned',
          options=D2HBenchmarkOptions(
              approach=JAX_ASYNC, enable_pinned_host_transfer=[False, True]
          ),
          expected_len=2,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )

    benchmarks = benchmark.generate()

    self.assertLen(benchmarks, expected_len)
    for b in benchmarks:
      self.assertIsInstance(b.options, D2HBenchmarkOptions)

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_jax_async',
          approach=JAX_ASYNC,
          enable_pinned_host_transfer=False,
          expected_valid=True,
      ),
      dict(
          testcase_name='valid_torch_dcp_style',
          approach=TORCH_DCP_STYLE,
          enable_pinned_host_transfer=False,
          expected_valid=True,
      ),
      dict(
          testcase_name='valid_jax_async_with_pinned',
          approach=JAX_ASYNC,
          enable_pinned_host_transfer=True,
          expected_valid=True,
      ),
      dict(
          testcase_name='invalid_unknown_approach',
          approach='unknown_approach',
          enable_pinned_host_transfer=False,
          expected_valid=False,
      ),
      dict(
          testcase_name='invalid_pinned_with_torch_dcp_style',
          approach=TORCH_DCP_STYLE,
          enable_pinned_host_transfer=True,
          expected_valid=False,
      ),
  )
  def test_options_is_valid(
      self, approach, enable_pinned_host_transfer, expected_valid
  ):
    options = D2HBenchmarkOptions(
        approach=approach,
        enable_pinned_host_transfer=enable_pinned_host_transfer,
    )
    self.assertEqual(options.is_valid(), expected_valid)

  def test_options_class_is_d2h_benchmark_options(self):
    self.assertEqual(D2HBenchmarkOptions, D2HBenchmark.options_class)

  def test_non_array_leaves_are_ignored(self):
    """Non-JAX-array pytree leaves (e.g. scalars) must not be transferred."""
    options = D2HBenchmarkOptions(approach=JAX_ASYNC)
    benchmark = D2HBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    mixed_pytree = {
        'array': jnp.ones((4, 4)),
        'scalar': 42,
        'string': 'hello',
    }
    context = benchmarks_core.TestContext(
        pytree=mixed_pytree,
        path=self.directory,
        options=options,
    )
    with mock.patch.object(
        d2h_benchmark, 'jax_async_d2h', wraps=d2h_benchmark.jax_async_d2h
    ) as mock_jax_d2h:
      benchmark.test_fn(context)

    called_arrays, _ = mock_jax_d2h.call_args
    self.assertLen(called_arrays[0], 1)
    self.assertIsInstance(called_arrays[0][0], jax.Array)


if __name__ == '__main__':
  absltest.main()

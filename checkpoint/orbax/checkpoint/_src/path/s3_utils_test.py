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

"""Tests for s3_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import s3_utils


class IsS3PathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', 's3://bucket/path', True),
      ('nested', 's3://bucket/a/b/c', True),
      ('no_path', 's3://bucket', True),
      ('trailing_slash', 's3://bucket/path/', True),
      ('gcs', 'gs://bucket/path', False),
      ('local', '/tmp/checkpoint', False),
      ('relative', 'checkpoint/dir', False),
  )
  def test_is_s3_path(self, path_str, expected):
    path = epath.Path(path_str)
    self.assertEqual(s3_utils.is_s3_path(path), expected)


class ParseS3PathTest(parameterized.TestCase):

  def test_simple_path(self):
    bucket, obj_path = s3_utils.parse_s3_path('s3://my-bucket/my/object')
    self.assertEqual(bucket, 'my-bucket')
    self.assertEqual(obj_path, 'my/object')

  def test_nested_path(self):
    bucket, obj_path = s3_utils.parse_s3_path('s3://bucket/a/b/c/d.txt')
    self.assertEqual(bucket, 'bucket')
    self.assertEqual(obj_path, 'a/b/c/d.txt')

  def test_no_object_path(self):
    bucket, obj_path = s3_utils.parse_s3_path('s3://bucket')
    self.assertEqual(bucket, 'bucket')
    self.assertEqual(obj_path, '')

  def test_trailing_slash(self):
    bucket, obj_path = s3_utils.parse_s3_path('s3://bucket/path/')
    self.assertEqual(bucket, 'bucket')
    self.assertEqual(obj_path, 'path/')

  def test_wrong_scheme_raises(self):
    with self.assertRaises(AssertionError):
      s3_utils.parse_s3_path('gs://bucket/path')

  def test_epath_path_input(self):
    path = epath.Path('s3://my-bucket/checkpoint/step_0')
    bucket, obj_path = s3_utils.parse_s3_path(path)
    self.assertEqual(bucket, 'my-bucket')
    self.assertEqual(obj_path, 'checkpoint/step_0')


if __name__ == '__main__':
  absltest.main()

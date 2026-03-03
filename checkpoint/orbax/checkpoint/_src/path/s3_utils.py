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

"""Utils for interacting with S3 paths."""

from urllib import parse
from etils import epath


_S3_PATH_PREFIX = 's3://'


def is_s3_path(path: epath.Path) -> bool:
  return path.as_posix().startswith(_S3_PATH_PREFIX)


def parse_s3_path(path: epath.PathLike) -> tuple[str, str]:
  parsed = parse.urlparse(str(path))
  assert parsed.scheme == 's3', f'Unsupported scheme for S3: {parsed.scheme}'
  bucket = parsed.netloc
  # Strip leading slash from path component.
  obj_path = parsed.path
  if obj_path.startswith('/'):
    obj_path = obj_path[1:]
  return bucket, obj_path

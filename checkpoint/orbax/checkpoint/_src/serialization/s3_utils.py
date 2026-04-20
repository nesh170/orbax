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

"""Boto3-based S3 read/write helpers for Orbax checkpointing."""

import os
from pathlib import Path
from typing import Optional

from absl import logging


def is_s3_path(path: str) -> bool:
  return path.startswith('s3://')


def parse_s3_path(s3_uri: str) -> tuple[str, str]:
  """Returns (bucket, key_prefix) from an s3:// URI."""
  without_scheme = s3_uri[len('s3://'):]
  bucket, _, prefix = without_scheme.partition('/')
  return bucket, prefix


def upload_directory_to_s3(
    local_dir: str,
    s3_uri: str,
    s3_client=None,
) -> None:
  """Upload all files under local_dir to s3_uri using boto3.

  The local directory tree is mirrored under s3_uri so that
  ``local_dir/a/b`` becomes ``s3_uri/a/b``.

  Args:
    local_dir: Absolute path to the local directory to upload.
    s3_uri: Destination S3 URI (e.g. ``s3://bucket/prefix``).
    s3_client: Optional pre-constructed boto3 S3 client.  A new client is
      created if not provided.
  """
  import boto3  # pylint: disable=g-import-not-at-top

  bucket, prefix = parse_s3_path(s3_uri)
  client = s3_client or boto3.client('s3')
  local_root = Path(local_dir)
  uploaded = 0
  for local_file in sorted(local_root.rglob('*')):
    if not local_file.is_file():
      continue
    relative = local_file.relative_to(local_root)
    s3_key = '/'.join([prefix.rstrip('/'), str(relative)]) if prefix else str(relative)
    logging.vlog(1, 'S3 upload: %s -> s3://%s/%s', local_file, bucket, s3_key)
    client.upload_file(str(local_file), bucket, s3_key)
    uploaded += 1
  logging.info('Uploaded %d file(s) to %s', uploaded, s3_uri)


def download_directory_from_s3(
    s3_uri: str,
    local_dir: str,
    s3_client=None,
) -> None:
  """Download all objects under s3_uri to local_dir using boto3.

  The S3 key structure relative to ``s3_uri`` is reproduced under
  ``local_dir``.

  Args:
    s3_uri: Source S3 URI (e.g. ``s3://bucket/prefix``).
    local_dir: Absolute path to the local directory to populate.
    s3_client: Optional pre-constructed boto3 S3 client.  A new client is
      created if not provided.
  """
  import boto3  # pylint: disable=g-import-not-at-top

  bucket, prefix = parse_s3_path(s3_uri)
  client = s3_client or boto3.client('s3')
  paginator = client.get_paginator('list_objects_v2')
  prefix_with_slash = prefix.rstrip('/') + '/' if prefix else ''
  downloaded = 0
  for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
    for obj in page.get('Contents', []):
      key = obj['Key']
      # Strip the common prefix so the local path mirrors the remote structure.
      relative = key[len(prefix_with_slash):]
      if not relative:
        continue
      local_file = Path(local_dir) / relative
      local_file.parent.mkdir(parents=True, exist_ok=True)
      logging.vlog(
          1, 'S3 download: s3://%s/%s -> %s', bucket, key, local_file
      )
      client.download_file(bucket, key, str(local_file))
      downloaded += 1
  logging.info('Downloaded %d file(s) from %s', downloaded, s3_uri)

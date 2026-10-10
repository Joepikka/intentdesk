import assert from 'node:assert/strict';
import test from 'node:test';
import {clampQueueOffset} from '../src/intentdesk/web/queue.mjs';

test('reviewing the last item returns to an existing page', () => {
  assert.equal(clampQueueOffset(20, 21, 20), 20);
  assert.equal(clampQueueOffset(20, 20, 20), 0);
  assert.equal(clampQueueOffset(40, 40, 20), 20);
});
test('empty queues reset to zero and valid pages stay put', () => {
  assert.equal(clampQueueOffset(20, 0, 20), 0);
  assert.equal(clampQueueOffset(0, 100, 20), 0);
  assert.equal(clampQueueOffset(20, 100, 20), 20);
});

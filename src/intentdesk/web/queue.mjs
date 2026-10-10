// Keep a queue page valid after its last pending request is reviewed.
export function clampQueueOffset(offset, total, limit) {
  return Math.min(offset, Math.floor(Math.max(0, total - 1) / limit) * limit);
}

// SPDX-License-Identifier: Apache-2.0
// Test stand-in for src/backends/transformersjs_worker.mjs: one float32 1x4 input echoed to one output, and no ONNX Runtime
// log on stderr, so a device=webgpu load must be refused as not placed on WebGPU. Needs Node only.
let pending = Buffer.alloc(0);
const send = (lines, payload = Buffer.alloc(0)) => {
  const header = Buffer.from(lines.join('\n'));
  const prefix = Buffer.alloc(8);
  prefix.writeUInt32LE(header.length, 0);
  prefix.writeUInt32LE(payload.length, 4);
  process.stdout.write(Buffer.concat([prefix, header, payload]));
};
process.stdin.on('data', (chunk) => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= 8) {
    const h = pending.readUInt32LE(0);
    const p = pending.readUInt32LE(4);
    if (pending.length < 8 + h + p) break;
    const command = pending.subarray(8, 8 + h).toString().split('\n')[0];
    const payload = pending.subarray(8 + h, 8 + h + p);
    pending = pending.subarray(8 + h + p);
    if (command === 'load') send(['ok', 'n_inputs=1', 'input0=x|float32|1x4', 'n_outputs=1', 'output0=y|float32|1x4']);
    else if (command === 'run') send(['ok'], payload);
    else process.exit(0);
  }
});

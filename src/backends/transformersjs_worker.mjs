// SPDX-License-Identifier: Apache-2.0
// Worker for the transformersjs backend (src/backends/transformersjs.c). It runs Transformers.js in Node and speaks a small
// binary protocol over stdin and stdout: a frame is u32le header length, u32le payload length, the header (text lines, the first
// is the command or the status, the others key=value) and the payload (raw tensor bytes). Anything else goes to stderr.
//
//   node transformersjs_worker.mjs <directory that resolves @huggingface/transformers>
//
// load  path= device= dtype= shapes=name:1x16;name:1x16   -> ok n_inputs= input<i>=name|dtype|dims n_outputs= output<i>=name|dtype|dims
// run   (payload: the inputs, in order)                    -> ok, payload: the outputs, in order
// quit
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

console.log = console.error;
const TYPES = {
  float32: [Float32Array, 4],
  int64: [BigInt64Array, 8],
  int32: [Int32Array, 4],
  int8: [Int8Array, 1],
  uint8: [Uint8Array, 1],
  bool: [Uint8Array, 1],
};

let tf;
let model;
let inputs = [];
let outputs = [];

const elements = (dims) => dims.reduce((a, b) => a * b, 1);
const parseDims = (text) => text.split('x').map(Number);

function reply(status, lines = [], payload = Buffer.alloc(0)) {
  const header = Buffer.from([status, ...lines].join('\n'), 'utf8');
  const prefix = Buffer.alloc(8);
  prefix.writeUInt32LE(header.length, 0);
  prefix.writeUInt32LE(payload.length, 4);
  process.stdout.write(Buffer.concat([prefix, header, payload]));
}

const fail = (message, code = 'incompatible') =>
  reply('err', ['code=' + code, 'message=' + String(message).replace(/\s+/g, ' ').slice(0, 800)]);
const problem = (message, code) => Object.assign(new Error(message), { code });

function parseShapes(text) {
  const shapes = {};
  for (const entry of text.split(';').filter(Boolean)) {
    const at = entry.lastIndexOf(':');
    if (at < 0) throw new Error(`input_shapes entry '${entry}' is not name:1x2x3`);
    shapes[entry.slice(0, at)] = parseDims(entry.slice(at + 1));
  }
  return shapes;
}

function makeTensor(spec, bytes) {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new tf.Tensor(spec.dtype, new TYPES[spec.dtype][0](copy.buffer), spec.dims);
}

async function load(h) {
  const require = createRequire(h.package_dir.replace(/\/?$/, '/'));
  let entry;
  try {
    entry = require.resolve('@huggingface/transformers');
  } catch {
    throw problem(`@huggingface/transformers is not installed in ${h.package_dir} (npm install it there, or set package_dir)`, 'unavailable');
  }
  tf = await import(pathToFileURL(entry).href);
  const version = JSON.parse(readFileSync(new URL('../package.json', pathToFileURL(entry)), 'utf8')).version;
  tf.env.allowRemoteModels = false;
  model = await tf.AutoModel.from_pretrained(h.path, {
    device: h.device,
    dtype: h.dtype || 'fp32',
    // verbose, so the plugin can read the node placement from stderr and refuse a device the model did not land on
    session_options: { logSeverityLevel: 0 },
  });
  const names = Object.keys(model.sessions || {});
  if (names.length !== 1) throw problem(`the model has ${names.length} ONNX sessions (${names.join(', ')}); one is supported`, 'unsupported');
  const session = model.sessions[names[0]];
  const given = parseShapes(h.shapes || '');
  inputs = session.inputMetadata.map((meta) => {
    if (!TYPES[meta.type]) throw problem(`input '${meta.name}' has type ${meta.type}, which is not supported`, 'unsupported');
    const dims = given[meta.name] || meta.shape;
    if (!dims.every((d) => Number.isInteger(d) && d > 0))
      throw problem(`input '${meta.name}' has a dynamic dimension (${meta.shape.join('x')}); pass input_shapes=${meta.name}:1x16`, 'invalid_argument');
    return { name: meta.name, dtype: meta.type, dims };
  });
  const feeds = {};
  for (const spec of inputs) feeds[spec.name] = makeTensor(spec, new Uint8Array(elements(spec.dims) * TYPES[spec.dtype][1]));
  const result = await model(feeds);
  outputs = session.outputNames.map((name) => {
    const tensor = result[name];
    if (!tensor) throw new Error(`the model did not return output '${name}'`);
    if (!TYPES[tensor.type]) throw problem(`output '${name}' has type ${tensor.type}, which is not supported`, 'unsupported');
    return { name, dtype: tensor.type, dims: Array.from(tensor.dims) };
  });
  const describe = (specs, label) => specs.map((s, i) => `${label}${i}=${s.name}|${s.dtype}|${s.dims.join('x')}`);
  reply('ok', [
    `transformers=${version}`,
    `n_inputs=${inputs.length}`,
    ...describe(inputs, 'input'),
    `n_outputs=${outputs.length}`,
    ...describe(outputs, 'output'),
  ]);
}

async function run(payload) {
  const feeds = {};
  let offset = 0;
  for (const spec of inputs) {
    const nbytes = elements(spec.dims) * TYPES[spec.dtype][1];
    feeds[spec.name] = makeTensor(spec, payload.subarray(offset, offset + nbytes));
    offset += nbytes;
  }
  const result = await model(feeds);
  const parts = outputs.map((spec) => {
    const data = result[spec.name].data;
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  });
  reply('ok', [], Buffer.concat(parts));
}

function parseHeader(buffer) {
  const [command, ...rest] = buffer.toString('utf8').split('\n');
  const h = { command };
  for (const line of rest) {
    const at = line.indexOf('=');
    if (at > 0) h[line.slice(0, at)] = line.slice(at + 1);
  }
  return h;
}

let pending = Buffer.alloc(0);
let queue = Promise.resolve();
process.stdin.on('data', (chunk) => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= 8) {
    const headerLength = pending.readUInt32LE(0);
    const payloadLength = pending.readUInt32LE(4);
    if (pending.length < 8 + headerLength + payloadLength) break;
    const header = parseHeader(pending.subarray(8, 8 + headerLength));
    const payload = pending.subarray(8 + headerLength, 8 + headerLength + payloadLength);
    pending = pending.subarray(8 + headerLength + payloadLength);
    queue = queue.then(async () => {
      try {
        if (header.command === 'load') await load(header);
        else if (header.command === 'run') await run(payload);
        else if (header.command === 'quit') process.exit(0);
        else fail(`unknown command '${header.command}'`);
      } catch (error) {
        fail(error && error.message ? error.message : error, error && error.code);
      }
    });
  }
});
process.stdin.on('end', () => process.exit(0));

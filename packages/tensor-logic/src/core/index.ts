/**
 * Tensor Logic Core Module
 *
 * Exports all core tensor types, operations, and nonlinearities
 */

// Nonlinearities
export {
  abs,
  batchNorm,
  clamp,
  cos,
  dropout,
  elu,
  exp,
  gelu,
  leakyRelu,
  lnorm,
  log,
  logSoftmax,
  neg,
  pow,
  prelu,
  reciprocal,
  relu,
  selu,
  sigmoid,
  sigmoidTemperature,
  sin,
  smoothStep,
  softmax,
  sqrt,
  square,
  step,
  swish,
  tanh,
} from './nonlinearities'

// Operations
export {
  add,
  argmax,
  argmin,
  avgReduce,
  concat,
  divide,
  einsum,
  hadamard,
  join,
  max,
  maxReduce,
  mean,
  min,
  norm,
  normalize,
  outer,
  project,
  scale,
  slice,
  subtract,
  sum,
  transpose,
} from './operations'

// Types
export type {
  DenseTensor,
  IndexName,
  SparseEntry,
  SparseTensor,
  Tensor,
  TensorDType,
  TensorIndex,
  TensorShape,
} from './types'

export {
  clone,
  coordsToFlat,
  createDenseTensor,
  createEmbedding,
  createRelation,
  createShape,
  createSparseTensor,
  dims,
  eye,
  flatToCoords,
  full,
  getElement,
  ones,
  randn,
  random,
  rank,
  reshape,
  setElement,
  toDense,
  toSparse,
  zeros,
} from './types'

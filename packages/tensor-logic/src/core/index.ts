/**
 * Tensor Logic Core Module
 *
 * Exports all core tensor types, operations, and nonlinearities
 */

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
  random,
  randn,
  rank,
  reshape,
  setElement,
  toDense,
  toSparse,
  zeros,
} from './types'

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

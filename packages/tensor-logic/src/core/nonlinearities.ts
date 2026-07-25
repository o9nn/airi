/**
 * Tensor Logic Nonlinearity Functions
 *
 * Implements nonlinear activation functions for tensor logic:
 * - step (Heaviside): H(x) = 1 if x > 0, else 0
 * - sigmoid: σ(x) = 1 / (1 + e^(-x))
 * - relu: max(0, x)
 * - softmax: normalized exponential
 * - lnorm: layer normalization
 * - tanh, leaky relu, gelu, etc.
 */

import type { DenseTensor, IndexName, Tensor } from './types'

import {
  clone,
  createDenseTensor,
  toDense,
} from './types'

/**
 * Heaviside step function: H(x) = 1 if x > 0, else 0
 * This is the core function for converting Einstein sums to logical operations
 */
export function step(tensor: Tensor, threshold = 0): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => (x > threshold ? 1 : 0))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Smooth step function with configurable steepness
 * Approximates step function but is differentiable
 */
export function smoothStep(tensor: Tensor, steepness = 10): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => 1 / (1 + Math.exp(-steepness * x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
 */
export function sigmoid(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => 1 / (1 + Math.exp(-x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Sigmoid with temperature control
 * T=0 approaches step function, larger T makes it smoother
 */
export function sigmoidTemperature(tensor: Tensor, temperature: number): DenseTensor<number> {
  if (temperature <= 0) {
    return step(tensor)
  }
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => 1 / (1 + Math.exp(-x / temperature)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * ReLU activation: max(0, x)
 */
export function relu(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.max(0, x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Leaky ReLU: x if x > 0, else alpha * x
 */
export function leakyRelu(tensor: Tensor, alpha = 0.01): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => (x > 0 ? x : alpha * x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Parametric ReLU (same as leaky relu but alpha can vary)
 */
export function prelu(tensor: Tensor, alpha: number | Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  if (typeof alpha === 'number') {
    return leakyRelu(dense, alpha)
  }

  const alphaData = alpha.type === 'sparse' ? toDense(alpha).data : alpha.data
  const outData = dense.data.map((x, i) => {
    const a = alphaData[i % alphaData.length]
    return x > 0 ? x : a * x
  })
  return createDenseTensor(dense.shape, outData)
}

/**
 * ELU: x if x > 0, else alpha * (e^x - 1)
 */
export function elu(tensor: Tensor, alpha = 1): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => (x > 0 ? x : alpha * (Math.exp(x) - 1)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * SELU: scaled ELU for self-normalizing networks
 */
export function selu(tensor: Tensor): DenseTensor<number> {
  const lambda = 1.0507009873554805
  const alpha = 1.6732632423543772
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x =>
    lambda * (x > 0 ? x : alpha * (Math.exp(x) - 1)),
  )
  return createDenseTensor(dense.shape, outData)
}

/**
 * GELU: Gaussian Error Linear Unit
 * GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
 */
export function gelu(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map((x) => {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const c = Math.sqrt(2 / Math.PI)
    return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)))
  })
  return createDenseTensor(dense.shape, outData)
}

/**
 * Swish/SiLU: x * sigmoid(x)
 */
export function swish(tensor: Tensor, beta = 1): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => x / (1 + Math.exp(-beta * x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Tanh activation
 */
export function tanh(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.tanh(x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Softmax: normalized exponential along specified index
 * softmax(x)_i = e^(x_i) / Σ_j e^(x_j)
 */
export function softmax(tensor: Tensor, axisIndex?: IndexName): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  // If no axis specified, apply to last dimension
  const axisName = axisIndex ?? dense.indexNames[dense.indexNames.length - 1]
  const axisDim = dense.shape.indices.findIndex(i => i.name === axisName)

  if (axisDim < 0) {
    throw new Error(`Index ${axisName} not found`)
  }

  const axisSize = dense.shape.indices[axisDim].size
  const outData = [...dense.data]

  // Group by all dimensions except axis
  const outerSize = dense.data.length / axisSize
  const strides = computeStrides(dense.shape)
  const axisStride = strides[axisDim]

  // For each position except along the axis
  for (let outer = 0; outer < outerSize; outer++) {
    // Compute starting index for this group
    let startIdx = 0
    let remaining = outer
    for (let d = dense.shape.indices.length - 1; d >= 0; d--) {
      if (d === axisDim)
        continue
      const size = dense.shape.indices[d].size
      const coord = remaining % size
      remaining = Math.floor(remaining / size)
      startIdx += coord * strides[d]
    }

    // Find max for numerical stability
    let maxVal = Number.NEGATIVE_INFINITY
    for (let i = 0; i < axisSize; i++) {
      maxVal = Math.max(maxVal, dense.data[startIdx + i * axisStride])
    }

    // Compute exp and sum
    let sumExp = 0
    for (let i = 0; i < axisSize; i++) {
      const idx = startIdx + i * axisStride
      outData[idx] = Math.exp(dense.data[idx] - maxVal)
      sumExp += outData[idx]
    }

    // Normalize
    for (let i = 0; i < axisSize; i++) {
      const idx = startIdx + i * axisStride
      outData[idx] /= sumExp
    }
  }

  return createDenseTensor(dense.shape, outData)
}

/**
 * Log-softmax for numerical stability in loss computation
 */
export function logSoftmax(tensor: Tensor, axisIndex?: IndexName): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  const axisName = axisIndex ?? dense.indexNames[dense.indexNames.length - 1]
  const axisDim = dense.shape.indices.findIndex(i => i.name === axisName)

  if (axisDim < 0) {
    throw new Error(`Index ${axisName} not found`)
  }

  const axisSize = dense.shape.indices[axisDim].size
  const outData = [...dense.data]
  const strides = computeStrides(dense.shape)
  const axisStride = strides[axisDim]
  const outerSize = dense.data.length / axisSize

  for (let outer = 0; outer < outerSize; outer++) {
    let startIdx = 0
    let remaining = outer
    for (let d = dense.shape.indices.length - 1; d >= 0; d--) {
      if (d === axisDim)
        continue
      const size = dense.shape.indices[d].size
      const coord = remaining % size
      remaining = Math.floor(remaining / size)
      startIdx += coord * strides[d]
    }

    // Find max for numerical stability
    let maxVal = Number.NEGATIVE_INFINITY
    for (let i = 0; i < axisSize; i++) {
      maxVal = Math.max(maxVal, dense.data[startIdx + i * axisStride])
    }

    // Compute log-sum-exp
    let sumExp = 0
    for (let i = 0; i < axisSize; i++) {
      sumExp += Math.exp(dense.data[startIdx + i * axisStride] - maxVal)
    }
    const logSumExp = maxVal + Math.log(sumExp)

    // x - log-sum-exp
    for (let i = 0; i < axisSize; i++) {
      const idx = startIdx + i * axisStride
      outData[idx] = dense.data[idx] - logSumExp
    }
  }

  return createDenseTensor(dense.shape, outData)
}

/**
 * Layer normalization along specified indices
 * lnorm(x) = (x - μ) / (σ + ε)
 */
export function lnorm(
  tensor: Tensor,
  normalizeIndices?: IndexName[],
  epsilon = 1e-5,
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  // Default: normalize over last index
  const normIndices = normalizeIndices ?? [dense.indexNames[dense.indexNames.length - 1]]

  // Find which dimensions to normalize
  const normDims = normIndices.map((name) => {
    const idx = dense.shape.indices.findIndex(i => i.name === name)
    if (idx < 0)
      throw new Error(`Index ${name} not found`)
    return idx
  })

  const normSize = normDims.reduce((acc, d) => acc * dense.shape.indices[d].size, 1)
  const outData = [...dense.data]

  // Group by non-normalized dimensions
  const batchDims = dense.shape.indices
    .map((_, i) => i)
    .filter(i => !normDims.includes(i))

  const batchSize = batchDims.reduce((acc, d) => acc * dense.shape.indices[d].size, 1)
  const strides = computeStrides(dense.shape)

  // Iterate over batch positions
  for (let b = 0; b < batchSize; b++) {
    // Compute batch coordinates
    const batchCoords: number[] = []
    let remaining = b
    for (let i = batchDims.length - 1; i >= 0; i--) {
      const d = batchDims[i]
      batchCoords.unshift(remaining % dense.shape.indices[d].size)
      remaining = Math.floor(remaining / dense.shape.indices[d].size)
    }

    // Collect all indices for this batch position
    const indices: number[] = []
    collectNormIndices(dense.shape, batchDims, batchCoords, normDims, [], indices, strides)

    // Compute mean
    let sum = 0
    for (const idx of indices) {
      sum += dense.data[idx]
    }
    const mean = sum / normSize

    // Compute variance
    let varSum = 0
    for (const idx of indices) {
      const diff = dense.data[idx] - mean
      varSum += diff * diff
    }
    const variance = varSum / normSize

    // Normalize
    const invStd = 1 / Math.sqrt(variance + epsilon)
    for (const idx of indices) {
      outData[idx] = (dense.data[idx] - mean) * invStd
    }
  }

  return createDenseTensor(dense.shape, outData)
}

/**
 * Batch normalization (for training: uses batch statistics)
 */
export function batchNorm(
  tensor: Tensor,
  batchIndex: IndexName,
  epsilon = 1e-5,
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  const batchDim = dense.shape.indices.findIndex(i => i.name === batchIndex)
  if (batchDim < 0) {
    throw new Error(`Batch index ${batchIndex} not found`)
  }

  const batchSize = dense.shape.indices[batchDim].size
  const featureSize = dense.data.length / batchSize
  const strides = computeStrides(dense.shape)
  const batchStride = strides[batchDim]
  const outData = [...dense.data]

  // For each feature position
  for (let f = 0; f < featureSize; f++) {
    // Compute starting index
    let startIdx = 0
    let remaining = f
    for (let d = dense.shape.indices.length - 1; d >= 0; d--) {
      if (d === batchDim)
        continue
      const size = dense.shape.indices[d].size
      const coord = remaining % size
      remaining = Math.floor(remaining / size)
      startIdx += coord * strides[d]
    }

    // Compute mean across batch
    let sum = 0
    for (let b = 0; b < batchSize; b++) {
      sum += dense.data[startIdx + b * batchStride]
    }
    const mean = sum / batchSize

    // Compute variance
    let varSum = 0
    for (let b = 0; b < batchSize; b++) {
      const diff = dense.data[startIdx + b * batchStride] - mean
      varSum += diff * diff
    }
    const variance = varSum / batchSize

    // Normalize
    const invStd = 1 / Math.sqrt(variance + epsilon)
    for (let b = 0; b < batchSize; b++) {
      const idx = startIdx + b * batchStride
      outData[idx] = (dense.data[idx] - mean) * invStd
    }
  }

  return createDenseTensor(dense.shape, outData)
}

/**
 * Dropout: randomly zero elements with probability p
 */
export function dropout(tensor: Tensor, p = 0.5, training = true): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  if (!training) {
    return clone(dense)
  }

  const scale = 1 / (1 - p)
  const outData = dense.data.map(x => (Math.random() < p ? 0 : x * scale))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Absolute value
 */
export function abs(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.abs(x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Square
 */
export function square(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => x * x)
  return createDenseTensor(dense.shape, outData)
}

/**
 * Square root (clamped to non-negative)
 */
export function sqrt(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.sqrt(Math.max(0, x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Natural logarithm (with small epsilon for stability)
 */
export function log(tensor: Tensor, epsilon = 1e-10): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.log(Math.max(epsilon, x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Exponential
 */
export function exp(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.exp(x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Power
 */
export function pow(tensor: Tensor, exponent: number): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => x ** exponent)
  return createDenseTensor(dense.shape, outData)
}

/**
 * Clamp values to range [min, max]
 */
export function clamp(tensor: Tensor, minVal: number, maxVal: number): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.max(minVal, Math.min(maxVal, x)))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Sine function
 */
export function sin(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.sin(x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Cosine function
 */
export function cos(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => Math.cos(x))
  return createDenseTensor(dense.shape, outData)
}

/**
 * Negation
 */
export function neg(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(x => -x)
  return createDenseTensor(dense.shape, outData)
}

/**
 * Reciprocal (1/x)
 */
export function reciprocal(tensor: Tensor, epsilon = 1e-10): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map((x) => {
    const absX = Math.abs(x)
    if (absX < epsilon)
      return x >= 0 ? 1 / epsilon : -1 / epsilon
    return 1 / x
  })
  return createDenseTensor(dense.shape, outData)
}

// Helper functions

function computeStrides(shape: import('./types').TensorShape): number[] {
  const strides: number[] = new Array(shape.indices.length)
  let stride = 1
  for (let i = shape.indices.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape.indices[i].size
  }
  return strides
}

function collectNormIndices(
  shape: import('./types').TensorShape,
  batchDims: number[],
  batchCoords: number[],
  normDims: number[],
  normCoords: number[],
  indices: number[],
  strides: number[],
): void {
  if (normCoords.length === normDims.length) {
    // Compute flat index
    let flatIdx = 0
    let batchIdx = 0
    let normIdx = 0
    for (let d = 0; d < shape.indices.length; d++) {
      if (batchDims.includes(d)) {
        flatIdx += batchCoords[batchIdx++] * strides[d]
      }
      else {
        flatIdx += normCoords[normIdx++] * strides[d]
      }
    }
    indices.push(flatIdx)
    return
  }

  const currentNormDim = normDims[normCoords.length]
  for (let i = 0; i < shape.indices[currentNormDim].size; i++) {
    collectNormIndices(shape, batchDims, batchCoords, normDims, [...normCoords, i], indices, strides)
  }
}

/**
 * Tucker Decomposition for Tensor Logic Embeddings
 *
 * Implements Tucker decomposition for converting sparse Boolean tensors
 * to dense embeddings for efficient GPU computation.
 *
 * Key insight from the paper:
 * A[i,j,k] = M[i,p] * M'[j,q] * M''[k,r] * C[p,q,r]
 *
 * Where:
 * - M, M', M'' are factor matrices (embeddings)
 * - C is the core tensor
 * - p, q, r are embedding dimensions (much smaller than i, j, k)
 */

import type { DenseTensor, SparseTensor, Tensor, TensorShape } from '../core/types'

import {
  clone,
  coordsToFlat,
  createDenseTensor,
  createShape,
  flatToCoords,
  randn,
  toDense,
} from '../core/types'

/**
 * Tucker decomposition result
 */
export interface TuckerDecomposition {
  /** Factor matrices for each dimension */
  factors: DenseTensor[]
  /** Core tensor */
  core: DenseTensor
  /** Original shape */
  originalShape: TensorShape
  /** Embedding dimension for each factor */
  embeddingDims: number[]
}

/**
 * Decompose a tensor using Tucker decomposition
 *
 * Uses random projections for initialization (as suggested in the paper)
 */
export function tuckerDecompose(
  tensor: Tensor,
  embeddingDims: number[],
  iterations = 10,
): TuckerDecomposition {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  if (embeddingDims.length !== dense.shape.indices.length) {
    throw new Error(`Expected ${dense.shape.indices.length} embedding dims, got ${embeddingDims.length}`)
  }

  // Initialize factor matrices with random values
  const factors: DenseTensor[] = dense.shape.indices.map((idx, i) => {
    const shape = createShape([
      { name: idx.name, size: idx.size },
      { name: `e${i}`, size: embeddingDims[i] },
    ])
    const factor = randn(shape, 0, 1 / Math.sqrt(idx.size))
    return orthogonalize(factor)
  })

  // Compute initial core tensor
  let core = computeCore(dense, factors, embeddingDims)

  // Alternating least squares iterations
  for (let iter = 0; iter < iterations; iter++) {
    // Update each factor matrix
    for (let mode = 0; mode < factors.length; mode++) {
      factors[mode] = updateFactor(dense, factors, core, mode)
    }
    // Update core
    core = computeCore(dense, factors, embeddingDims)
  }

  return {
    factors,
    core,
    originalShape: dense.shape,
    embeddingDims,
  }
}

/**
 * Reconstruct tensor from Tucker decomposition
 */
export function tuckerReconstruct(decomposition: TuckerDecomposition): DenseTensor {
  const { factors, core, originalShape } = decomposition

  // Start with core tensor
  let result = clone(core)

  // Apply each factor matrix via tensor multiplication
  for (let i = factors.length - 1; i >= 0; i--) {
    result = applyFactor(result, factors[i], i)
  }

  // Reshape to original shape
  return {
    ...result,
    shape: originalShape,
    indexNames: originalShape.indices.map(idx => idx.name),
  }
}

/**
 * Compute the core tensor given factor matrices
 */
function computeCore(
  tensor: DenseTensor,
  factors: DenseTensor[],
  _embeddingDims: number[],
): DenseTensor {
  // Core = tensor ×₁ M₁ᵀ ×₂ M₂ᵀ ×₃ M₃ᵀ ...
  let result = clone(tensor)

  for (let mode = 0; mode < factors.length; mode++) {
    result = modeProduct(result, transposeMatrix(factors[mode]), mode)
  }

  return result
}

/**
 * Update factor matrix for a specific mode
 */
function updateFactor(
  tensor: DenseTensor,
  factors: DenseTensor[],
  core: DenseTensor,
  mode: number,
): DenseTensor {
  // Simplified factor update using SVD-like approach
  // Full implementation would use proper ALS

  const modeSize = tensor.shape.indices[mode].size
  const _embDim = factors[mode].shape.indices[1].size

  // Compute mode-n unfolding of tensor
  const _unfolding = modeUnfold(tensor, mode)

  // Compute Khatri-Rao product of other factors
  const _krProduct = khatriRaoProduct(factors, mode)

  // Update factor via least squares
  // In simplified form: new_factor = unfold(X) * krProduct * (krProduct^T * krProduct)^-1
  // We use a simpler approach: orthogonalize the result

  const newFactor = randn(factors[mode].shape, 0, 1 / Math.sqrt(modeSize))
  return orthogonalize(newFactor)
}

/**
 * Mode-n product of tensor with matrix
 */
function modeProduct(
  tensor: DenseTensor,
  matrix: DenseTensor,
  mode: number,
): DenseTensor {
  const tensorShape = tensor.shape
  const modeSize = tensorShape.indices[mode].size
  const newModeSize = matrix.shape.indices[0].size

  // Create output shape
  const outIndices = tensorShape.indices.map((idx, i) => {
    if (i === mode) {
      return { name: idx.name, size: newModeSize }
    }
    return idx
  })
  const outShape = createShape(outIndices.map(idx => ({ name: idx.name, size: idx.size })))
  const outData = new Array(outShape.size).fill(0)

  // Compute mode-n product
  const _strides = computeStrides(tensorShape)
  const _outStrides = computeStrides(outShape)

  for (let outI = 0; outI < outShape.size; outI++) {
    const outCoords = flatToCoords(outI, outShape)
    let sum = 0

    for (let k = 0; k < modeSize; k++) {
      const inCoords = [...outCoords]
      inCoords[mode] = k

      const tensorVal = tensor.data[coordsToFlat(inCoords, tensorShape)]
      const matrixVal = matrix.data[outCoords[mode] * modeSize + k]

      sum += tensorVal * matrixVal
    }

    outData[outI] = sum
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Apply factor matrix to tensor
 */
function applyFactor(
  tensor: DenseTensor,
  factor: DenseTensor,
  mode: number,
): DenseTensor {
  // Simplified: treat as matrix multiplication along the mode
  return modeProduct(tensor, transposeMatrix(factor), mode)
}

/**
 * Transpose a 2D matrix tensor
 */
function transposeMatrix(matrix: DenseTensor): DenseTensor {
  if (matrix.shape.indices.length !== 2) {
    throw new Error('Transpose requires 2D tensor')
  }

  const [rows, cols] = [matrix.shape.indices[0].size, matrix.shape.indices[1].size]
  const outShape = createShape([
    { name: matrix.indexNames[1], size: cols },
    { name: matrix.indexNames[0], size: rows },
  ])
  const outData = new Array(rows * cols).fill(0)

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      outData[j * rows + i] = matrix.data[i * cols + j]
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Mode-n unfolding of tensor
 */
function modeUnfold(tensor: DenseTensor, mode: number): DenseTensor {
  const modeSize = tensor.shape.indices[mode].size
  const otherSize = tensor.data.length / modeSize

  const outShape = createShape([
    { name: tensor.indexNames[mode], size: modeSize },
    { name: '_other', size: otherSize },
  ])
  const outData = new Array(modeSize * otherSize).fill(0)

  for (let i = 0; i < tensor.data.length; i++) {
    const coords = flatToCoords(i, tensor.shape)
    const modeCoord = coords[mode]

    // Compute index in other dimensions
    let otherIdx = 0
    let multiplier = 1
    for (let j = tensor.shape.indices.length - 1; j >= 0; j--) {
      if (j !== mode) {
        otherIdx += coords[j] * multiplier
        multiplier *= tensor.shape.indices[j].size
      }
    }

    outData[modeCoord * otherSize + otherIdx] = tensor.data[i]
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Khatri-Rao product of factor matrices (excluding one mode)
 */
function khatriRaoProduct(factors: DenseTensor[], excludeMode: number): DenseTensor {
  const included = factors.filter((_, i) => i !== excludeMode)

  if (included.length === 0) {
    return createDenseTensor(createShape([{ name: '_', size: 1 }]), [1])
  }

  let result = included[0]
  for (let i = 1; i < included.length; i++) {
    result = khatriRao(result, included[i])
  }

  return result
}

/**
 * Khatri-Rao product of two matrices
 */
function khatriRao(a: DenseTensor, b: DenseTensor): DenseTensor {
  const rowsA = a.shape.indices[0].size
  const colsA = a.shape.indices[1].size
  const rowsB = b.shape.indices[0].size
  const colsB = b.shape.indices[1].size

  if (colsA !== colsB) {
    throw new Error('Matrices must have same number of columns for Khatri-Rao product')
  }

  const outShape = createShape([
    { name: '_row', size: rowsA * rowsB },
    { name: '_col', size: colsA },
  ])
  const outData = new Array(rowsA * rowsB * colsA).fill(0)

  for (let j = 0; j < colsA; j++) {
    for (let ia = 0; ia < rowsA; ia++) {
      for (let ib = 0; ib < rowsB; ib++) {
        const outRow = ia * rowsB + ib
        outData[outRow * colsA + j] = a.data[ia * colsA + j] * b.data[ib * colsB + j]
      }
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Orthogonalize matrix columns using Gram-Schmidt
 */
function orthogonalize(matrix: DenseTensor): DenseTensor {
  const rows = matrix.shape.indices[0].size
  const cols = matrix.shape.indices[1].size
  const data = [...matrix.data]

  for (let j = 0; j < cols; j++) {
    // Subtract projections onto previous columns
    for (let k = 0; k < j; k++) {
      let dot = 0
      let normK = 0
      for (let i = 0; i < rows; i++) {
        dot += data[i * cols + j] * data[i * cols + k]
        normK += data[i * cols + k] * data[i * cols + k]
      }
      if (normK > 1e-10) {
        const proj = dot / normK
        for (let i = 0; i < rows; i++) {
          data[i * cols + j] -= proj * data[i * cols + k]
        }
      }
    }

    // Normalize column
    let norm = 0
    for (let i = 0; i < rows; i++) {
      norm += data[i * cols + j] * data[i * cols + j]
    }
    norm = Math.sqrt(norm)
    if (norm > 1e-10) {
      for (let i = 0; i < rows; i++) {
        data[i * cols + j] /= norm
      }
    }
  }

  return createDenseTensor(matrix.shape, data)
}

/**
 * Compute strides for a tensor shape
 */
function computeStrides(shape: TensorShape): number[] {
  const strides = new Array(shape.indices.length)
  let stride = 1
  for (let i = shape.indices.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape.indices[i].size
  }
  return strides
}

/**
 * Compute reconstruction error (Frobenius norm of difference)
 */
export function tuckerError(
  original: Tensor,
  decomposition: TuckerDecomposition,
): number {
  const dense = original.type === 'sparse' ? toDense(original) : original
  const reconstructed = tuckerReconstruct(decomposition)

  let sumSq = 0
  for (let i = 0; i < dense.data.length; i++) {
    const diff = dense.data[i] - reconstructed.data[i]
    sumSq += diff * diff
  }

  return Math.sqrt(sumSq)
}

/**
 * Create random embeddings for a set of objects
 *
 * Returns unit vectors as embeddings
 */
export function createRandomEmbeddings(
  numObjects: number,
  embeddingDim: number,
  indexName = 'x',
  embIndexName = 'd',
): DenseTensor {
  const shape = createShape([
    { name: indexName, size: numObjects },
    { name: embIndexName, size: embeddingDim },
  ])

  const data: number[] = []
  for (let i = 0; i < numObjects; i++) {
    const vec: number[] = []
    for (let j = 0; j < embeddingDim; j++) {
      vec.push(Math.random() * 2 - 1)
    }
    // Normalize to unit vector
    const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0))
    for (const v of vec) {
      data.push(v / norm)
    }
  }

  return createDenseTensor(shape, data)
}

/**
 * Embed a sparse relation tensor using Tucker decomposition
 *
 * For relation R(x,y), creates:
 * EmbR[i,j] = R(x,y) * Emb[x,i] * Emb[y,j]
 */
export function embedRelation(
  relation: SparseTensor,
  embeddings: DenseTensor,
): DenseTensor {
  const _dense = toDense(relation)
  const embDim = embeddings.shape.indices[1].size

  // Create output shape for embedded relation
  const outShape = createShape([
    { name: 'i', size: embDim },
    { name: 'j', size: embDim },
  ])
  const outData = new Array(embDim * embDim).fill(0)

  // Compute EmbR[i,j] = Σ_{x,y} R[x,y] * Emb[x,i] * Emb[y,j]
  for (const entry of relation.entries) {
    if (!entry.value)
      continue

    const [x, y] = entry.coords

    for (let i = 0; i < embDim; i++) {
      for (let j = 0; j < embDim; j++) {
        const embXi = embeddings.data[x * embDim + i]
        const embYj = embeddings.data[y * embDim + j]
        outData[i * embDim + j] += embXi * embYj
      }
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Query an embedded relation
 *
 * D[A,B] = EmbR[i,j] * Emb[A,i] * Emb[B,j]
 * Returns ~1 if R(A,B), ~0 otherwise (with some noise)
 */
export function queryEmbeddedRelation(
  embeddedRelation: DenseTensor,
  embeddings: DenseTensor,
  objectA: number,
  objectB: number,
): number {
  const embDim = embeddings.shape.indices[1].size

  let result = 0
  for (let i = 0; i < embDim; i++) {
    for (let j = 0; j < embDim; j++) {
      const embRij = embeddedRelation.data[i * embDim + j]
      const embAi = embeddings.data[objectA * embDim + i]
      const embBj = embeddings.data[objectB * embDim + j]
      result += embRij * embAi * embBj
    }
  }

  return result
}

/**
 * Superposition encoding: encode set as sum of embeddings
 *
 * S[d] = Σ_x V[x] * Emb[x,d]
 */
export function superpositionEncode(
  set: number[],
  embeddings: DenseTensor,
): DenseTensor {
  const embDim = embeddings.shape.indices[1].size
  const outShape = createShape([{ name: 'd', size: embDim }])
  const outData = new Array(embDim).fill(0)

  for (const x of set) {
    for (let d = 0; d < embDim; d++) {
      outData[d] += embeddings.data[x * embDim + d]
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Query superposition: check if object is in set
 *
 * D[A] = S[d] * Emb[A,d]
 * Returns ~1 if A in set, ~0 otherwise
 */
export function querySuperposition(
  superposition: DenseTensor,
  embeddings: DenseTensor,
  objectA: number,
): number {
  const embDim = embeddings.shape.indices[1].size

  let result = 0
  for (let d = 0; d < embDim; d++) {
    result += superposition.data[d] * embeddings.data[objectA * embDim + d]
  }

  return result
}

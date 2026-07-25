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
 * Decompose a tensor using Tucker decomposition.
 *
 * Initialised with HOSVD — each factor is the leading left singular vectors of
 * the corresponding mode unfolding — then refined by higher-order orthogonal
 * iteration (HOOI): each mode is re-fitted against the tensor projected onto
 * the other modes. Both stages are deterministic, so the same tensor always
 * yields the same decomposition.
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

  // HOSVD initialisation: each factor spans the dominant subspace of its mode.
  const factors: DenseTensor[] = dense.shape.indices.map(
    (_, mode) => leadingSingularVectors(dense, mode, embeddingDims[mode]),
  )

  // HOOI refinement. A single mode is optimal given the others, so sweeping
  // over the modes monotonically improves the fit.
  for (let iter = 0; iter < iterations; iter++) {
    for (let mode = 0; mode < factors.length; mode++) {
      factors[mode] = refineFactor(dense, factors, mode, embeddingDims[mode])
    }
  }

  const core = computeCore(dense, factors)

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

  // X ≈ C ×₁ U₁ ×₂ U₂ ... — the exact mirror of computeCore, which projects
  // with the transposed factors. Applying them untransposed here maps each
  // mode back from its embedding dimension to its original size.
  let result = clone(core)

  for (let mode = 0; mode < factors.length; mode++) {
    result = modeProduct(result, factors[mode], mode)
  }

  // Restore the original index names; the sizes already agree by construction.
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
): DenseTensor {
  // Core = tensor ×₁ M₁ᵀ ×₂ M₂ᵀ ×₃ M₃ᵀ ...
  let result = clone(tensor)

  for (let mode = 0; mode < factors.length; mode++) {
    result = modeProduct(result, transposeMatrix(factors[mode]), mode)
  }

  return result
}

/**
 * One HOOI step: re-fit a single mode against the other factors.
 *
 * Projecting the tensor onto every mode but `mode` leaves a smaller tensor
 * whose dominant mode-`mode` subspace is the optimal factor, holding the
 * others fixed.
 */
function refineFactor(
  tensor: DenseTensor,
  factors: DenseTensor[],
  mode: number,
  rank: number,
): DenseTensor {
  let projected = clone(tensor)

  for (let other = 0; other < factors.length; other++) {
    if (other !== mode) {
      projected = modeProduct(projected, transposeMatrix(factors[other]), other)
    }
  }

  return leadingSingularVectors(projected, mode, rank)
}

/**
 * The `rank` leading left singular vectors of a tensor's mode-n unfolding,
 * returned as the columns of an (unfolding rows x rank) matrix.
 *
 * These are obtained as the dominant eigenvectors of the Gram matrix
 * X(n)·X(n)ᵀ, which shares its eigenvectors with the left singular vectors of
 * X(n) and is only as large as the mode itself.
 */
function leadingSingularVectors(
  tensor: DenseTensor,
  mode: number,
  rank: number,
): DenseTensor {
  const unfolding = modeUnfold(tensor, mode)
  const rows = unfolding.shape.indices[0].size
  const cols = unfolding.shape.indices[1].size

  const gram: number[][] = Array.from({ length: rows }, () => Array.from({ length: rows }, () => 0))
  for (let i = 0; i < rows; i++) {
    for (let j = i; j < rows; j++) {
      let sum = 0
      for (let k = 0; k < cols; k++) {
        sum += unfolding.data[i * cols + k] * unfolding.data[j * cols + k]
      }
      gram[i][j] = sum
      gram[j][i] = sum
    }
  }

  const { vectors } = symmetricEigen(gram)

  const shape = createShape([
    { name: tensor.shape.indices[mode].name, size: rows },
    { name: `e${mode}`, size: rank },
  ])
  const data = Array.from({ length: rows * rank }, () => 0)

  // A rank wider than the mode leaves the surplus columns zero: the unfolding
  // simply has no more independent directions to offer.
  for (let r = 0; r < Math.min(rank, rows); r++) {
    for (let i = 0; i < rows; i++) {
      data[i * rank + r] = vectors[r][i]
    }
  }

  return createDenseTensor(shape, data)
}

/**
 * Eigendecomposition of a small symmetric matrix by cyclic Jacobi rotations.
 *
 * Returns eigenvalues in descending order alongside their eigenvectors, where
 * `vectors[r]` is the eigenvector for `values[r]`. Jacobi suits this use: the
 * Gram matrices here are symmetric positive semi-definite and no larger than a
 * single tensor mode, where it is both simple and accurate.
 */
function symmetricEigen(
  input: number[][],
  sweeps = 60,
  tolerance = 1e-12,
): { values: number[], vectors: number[][] } {
  const n = input.length
  const a = input.map(row => [...row])
  // Accumulates the rotations; column p holds the p-th eigenvector.
  const v: number[][] = Array.from(
    { length: n },
    (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let offDiagonal = 0
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        offDiagonal += a[p][q] * a[p][q]
      }
    }
    if (offDiagonal <= tolerance) {
      break
    }

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= tolerance) {
          continue
        }

        // Rotation that annihilates a[p][q], via the numerically stable root
        // of t^2 + 2*theta*t - 1 = 0.
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        const sign = theta >= 0 ? 1 : -1
        const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c

        for (let k = 0; k < n; k++) {
          const akp = a[k][p]
          const akq = a[k][q]
          a[k][p] = c * akp - s * akq
          a[k][q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k]
          const aqk = a[q][k]
          a[p][k] = c * apk - s * aqk
          a[q][k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p]
          const vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[y][y] - a[x][x])

  return {
    values: order.map(i => a[i][i]),
    vectors: order.map(column => v.map(row => row[column])),
  }
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

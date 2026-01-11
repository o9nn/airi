/**
 * Tensor Logic Core Operations
 *
 * Implements Einstein summation operations:
 * - Tensor join (product with implicit sum over shared indices)
 * - Tensor projection (sum over specified indices)
 * - Elementwise operations (Hadamard product, addition, etc.)
 */

import type {
  DenseTensor,
  IndexName,
  SparseTensor,
  Tensor,
  TensorIndex,
  TensorShape,
} from './types'
import {
  clone,
  coordsToFlat,
  createDenseTensor,
  createShape,
  flatToCoords,
  toDense,
} from './types'

/**
 * Tensor projection: sum over indices not in the target set
 * π_α(T) = Σ_β T[α,β] where β = indices not in α
 */
export function project(
  tensor: Tensor,
  targetIndices: IndexName[],
): DenseTensor<number> {
  // Convert sparse to dense for computation
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  // Find indices to project out
  const keepIndices = dense.shape.indices.filter(idx => targetIndices.includes(idx.name))
  const sumIndices = dense.shape.indices.filter(idx => !targetIndices.includes(idx.name))

  if (sumIndices.length === 0) {
    // No projection needed
    return clone(dense)
  }

  // Create output shape
  const outShape = createShape(keepIndices.map(idx => ({ name: idx.name, size: idx.size })))
  const outData = new Array(outShape.size).fill(0)

  // Iterate over all elements and accumulate
  for (let i = 0; i < dense.data.length; i++) {
    const coords = flatToCoords(i, dense.shape)

    // Extract coordinates for output indices
    const outCoords = keepIndices.map((idx) => {
      const srcIdx = dense.shape.indices.findIndex(si => si.name === idx.name)
      return coords[srcIdx]
    })

    const outFlatIdx = outCoords.length > 0 ? coordsToFlat(outCoords, outShape) : 0
    outData[outFlatIdx] += dense.data[i]
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Tensor join (Einstein summation): multiply tensors and sum over shared indices
 * (U ⨝ V)[α,γ] = Σ_β U[α,β] * V[β,γ]
 *
 * Where:
 * - α = indices unique to U
 * - γ = indices unique to V
 * - β = common indices (summed over)
 */
export function join(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  // Convert to dense for computation
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  // Find index categories
  const indicesA = denseA.shape.indices
  const indicesB = denseB.shape.indices

  const commonNames = new Set<IndexName>()
  const indexANames = new Set(indicesA.map(i => i.name))
  const indexBNames = new Set(indicesB.map(i => i.name))

  for (const name of indexANames) {
    if (indexBNames.has(name)) {
      commonNames.add(name)
    }
  }

  // Unique indices
  const uniqueA = indicesA.filter(i => !commonNames.has(i.name))
  const uniqueB = indicesB.filter(i => !commonNames.has(i.name))
  const common = indicesA.filter(i => commonNames.has(i.name))

  // Output shape: uniqueA + uniqueB
  const outIndices = [...uniqueA, ...uniqueB]
  const outShape = createShape(outIndices.map(idx => ({ name: idx.name, size: idx.size })))
  const outData = new Array(outShape.size).fill(0)

  // If no common indices, this is Kronecker product
  // If all indices common, this is Hadamard product
  if (common.length === 0) {
    // Kronecker product
    for (let i = 0; i < denseA.data.length; i++) {
      for (let j = 0; j < denseB.data.length; j++) {
        const coordsA = flatToCoords(i, denseA.shape)
        const coordsB = flatToCoords(j, denseB.shape)
        const outCoords = [...coordsA, ...coordsB]
        const outIdx = coordsToFlat(outCoords, outShape)
        outData[outIdx] = denseA.data[i] * denseB.data[j]
      }
    }
  }
  else {
    // General join with summation over common indices
    // Iterate over all combinations and accumulate

    // Build index maps
    const aIndexMap = new Map(indicesA.map((idx, i) => [idx.name, i]))
    const bIndexMap = new Map(indicesB.map((idx, i) => [idx.name, i]))

    // Compute sizes for iteration
    const commonSizes = common.map((idx) => {
      const aIdx = aIndexMap.get(idx.name)!
      return indicesA[aIdx].size
    })
    const commonTotal = commonSizes.reduce((a, b) => a * b, 1)

    const uniqueASizes = uniqueA.map(idx => idx.size)
    const uniqueATotal = uniqueASizes.reduce((a, b) => a * b, 1)

    const uniqueBSizes = uniqueB.map(idx => idx.size)
    const uniqueBTotal = uniqueBSizes.reduce((a, b) => a * b, 1)

    // Iterate over output coordinates
    for (let outI = 0; outI < outShape.size; outI++) {
      const outCoords = flatToCoords(outI, outShape)
      let sum = 0

      // Iterate over common indices
      for (let commonI = 0; commonI < commonTotal; commonI++) {
        // Compute common coordinates
        const commonCoords: number[] = []
        let remaining = commonI
        for (let k = common.length - 1; k >= 0; k--) {
          commonCoords.unshift(remaining % commonSizes[k])
          remaining = Math.floor(remaining / commonSizes[k])
        }

        // Build full coordinates for A
        const coordsA: number[] = new Array(indicesA.length)
        let uniqueAIdx = 0
        let commonIdx = 0
        for (let k = 0; k < indicesA.length; k++) {
          if (commonNames.has(indicesA[k].name)) {
            coordsA[k] = commonCoords[commonIdx++]
          }
          else {
            coordsA[k] = outCoords[uniqueAIdx++]
          }
        }

        // Build full coordinates for B
        const coordsB: number[] = new Array(indicesB.length)
        let uniqueBIdx = uniqueA.length
        commonIdx = 0
        for (let k = 0; k < indicesB.length; k++) {
          if (commonNames.has(indicesB[k].name)) {
            coordsB[k] = commonCoords[commonIdx++]
          }
          else {
            coordsB[k] = outCoords[uniqueBIdx++]
          }
        }

        const aFlatIdx = coordsToFlat(coordsA, denseA.shape)
        const bFlatIdx = coordsToFlat(coordsB, denseB.shape)

        sum += denseA.data[aFlatIdx] * denseB.data[bFlatIdx]
      }

      outData[outI] = sum
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Einstein summation notation: einsum('ij,jk->ik', A, B)
 * Parses einsum string and performs the operation
 */
export function einsum(
  notation: string,
  ...tensors: Tensor[]
): DenseTensor<number> {
  // Parse notation: 'ij,jk->ik' or 'ij,jk' (implicit output)
  const [inputPart, outputPart] = notation.split('->')

  if (!inputPart) {
    throw new Error('Invalid einsum notation: missing input indices')
  }

  const inputSpecs = inputPart.split(',').map(s => s.trim())

  if (inputSpecs.length !== tensors.length) {
    throw new Error(`Einsum: expected ${inputSpecs.length} tensors, got ${tensors.length}`)
  }

  // Rename tensor indices according to notation
  const renamedTensors: DenseTensor<number>[] = tensors.map((tensor, ti) => {
    const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
    const spec = inputSpecs[ti]

    if (spec.length !== dense.shape.indices.length) {
      throw new Error(`Einsum: tensor ${ti} has ${dense.shape.indices.length} indices, but notation specifies ${spec.length}`)
    }

    const newIndices = dense.shape.indices.map((idx, i) => ({
      ...idx,
      name: spec[i],
    }))

    return {
      ...dense,
      shape: { ...dense.shape, indices: newIndices },
      indexNames: newIndices.map(i => i.name),
    }
  })

  // Perform successive joins
  let result = renamedTensors[0]
  for (let i = 1; i < renamedTensors.length; i++) {
    result = join(result, renamedTensors[i])
  }

  // Project to output indices if specified
  if (outputPart) {
    const outputIndices = outputPart.trim().split('')
    result = project(result, outputIndices)
  }

  return result
}

/**
 * Hadamard (elementwise) product: tensors must have same shape
 */
export function hadamard(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  if (denseA.data.length !== denseB.data.length) {
    throw new Error('Hadamard product requires tensors of same size')
  }

  const outData = denseA.data.map((a, i) => a * denseB.data[i])
  return createDenseTensor(denseA.shape, outData)
}

/**
 * Elementwise addition
 */
export function add(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  // Handle scalar broadcasting
  if (denseB.data.length === 1 && denseA.data.length > 1) {
    const scalar = denseB.data[0]
    const outData = denseA.data.map(a => a + scalar)
    return createDenseTensor(denseA.shape, outData)
  }

  if (denseA.data.length === 1 && denseB.data.length > 1) {
    const scalar = denseA.data[0]
    const outData = denseB.data.map(b => scalar + b)
    return createDenseTensor(denseB.shape, outData)
  }

  if (denseA.data.length !== denseB.data.length) {
    throw new Error('Addition requires tensors of same size or scalar broadcasting')
  }

  const outData = denseA.data.map((a, i) => a + denseB.data[i])
  return createDenseTensor(denseA.shape, outData)
}

/**
 * Elementwise subtraction
 */
export function subtract(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  // Handle scalar broadcasting
  if (denseB.data.length === 1 && denseA.data.length > 1) {
    const scalar = denseB.data[0]
    const outData = denseA.data.map(a => a - scalar)
    return createDenseTensor(denseA.shape, outData)
  }

  if (denseA.data.length === 1 && denseB.data.length > 1) {
    const scalar = denseA.data[0]
    const outData = denseB.data.map(b => scalar - b)
    return createDenseTensor(denseB.shape, outData)
  }

  if (denseA.data.length !== denseB.data.length) {
    throw new Error('Subtraction requires tensors of same size or scalar broadcasting')
  }

  const outData = denseA.data.map((a, i) => a - denseB.data[i])
  return createDenseTensor(denseA.shape, outData)
}

/**
 * Scalar multiplication
 */
export function scale(
  tensor: Tensor,
  scalar: number,
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const outData = dense.data.map(v => v * scalar)
  return createDenseTensor(dense.shape, outData)
}

/**
 * Elementwise division
 */
export function divide(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  if (denseA.data.length !== denseB.data.length) {
    throw new Error('Division requires tensors of same size')
  }

  const outData = denseA.data.map((a, i) => {
    const b = denseB.data[i]
    return b === 0 ? 0 : a / b
  })
  return createDenseTensor(denseA.shape, outData)
}

/**
 * Outer product (Kronecker product): no shared indices
 */
export function outer(
  tensorA: Tensor,
  tensorB: Tensor,
): DenseTensor<number> {
  // For outer product, we need distinct index names
  const denseA = tensorA.type === 'sparse' ? toDense(tensorA) : tensorA
  const denseB = tensorB.type === 'sparse' ? toDense(tensorB) : tensorB

  // Check for overlapping indices
  const namesA = new Set(denseA.indexNames)
  const hasOverlap = denseB.indexNames.some(n => namesA.has(n))

  if (hasOverlap) {
    throw new Error('Outer product requires distinct index names')
  }

  return join(denseA, denseB)
}

/**
 * Matrix transpose (swap two indices)
 */
export function transpose(
  tensor: Tensor,
  idx1?: IndexName,
  idx2?: IndexName,
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  if (dense.shape.indices.length < 2) {
    throw new Error('Transpose requires at least 2D tensor')
  }

  // Default: swap first two indices
  const name1 = idx1 ?? dense.indexNames[0]
  const name2 = idx2 ?? dense.indexNames[1]

  const i1 = dense.shape.indices.findIndex(i => i.name === name1)
  const i2 = dense.shape.indices.findIndex(i => i.name === name2)

  if (i1 < 0 || i2 < 0) {
    throw new Error(`Index not found: ${name1} or ${name2}`)
  }

  // Create new shape with swapped indices
  const newIndices = [...dense.shape.indices]
  const temp = newIndices[i1]
  newIndices[i1] = newIndices[i2]
  newIndices[i2] = temp

  const newShape = createShape(newIndices.map(i => ({ name: i.name, size: i.size })))
  const newData = new Array(dense.data.length).fill(0)

  // Reorder data
  for (let i = 0; i < dense.data.length; i++) {
    const oldCoords = flatToCoords(i, dense.shape)
    const newCoords = [...oldCoords]
    const tmpCoord = newCoords[i1]
    newCoords[i1] = newCoords[i2]
    newCoords[i2] = tmpCoord
    const newFlatIdx = coordsToFlat(newCoords, newShape)
    newData[newFlatIdx] = dense.data[i]
  }

  return createDenseTensor(newShape, newData)
}

/**
 * Sum all elements of a tensor
 */
export function sum(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  return dense.data.reduce((a, b) => a + b, 0)
}

/**
 * Mean of all elements
 */
export function mean(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  return sum(dense) / dense.data.length
}

/**
 * Max element
 */
export function max(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  return Math.max(...dense.data)
}

/**
 * Min element
 */
export function min(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  return Math.min(...dense.data)
}

/**
 * Argmax: returns flat index of maximum element
 */
export function argmax(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  let maxIdx = 0
  let maxVal = dense.data[0]
  for (let i = 1; i < dense.data.length; i++) {
    if (dense.data[i] > maxVal) {
      maxVal = dense.data[i]
      maxIdx = i
    }
  }
  return maxIdx
}

/**
 * Argmin: returns flat index of minimum element
 */
export function argmin(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  let minIdx = 0
  let minVal = dense.data[0]
  for (let i = 1; i < dense.data.length; i++) {
    if (dense.data[i] < minVal) {
      minVal = dense.data[i]
      minIdx = i
    }
  }
  return minIdx
}

/**
 * L2 norm (Frobenius norm for matrices)
 */
export function norm(tensor: Tensor): number {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  return Math.sqrt(dense.data.reduce((acc, v) => acc + v * v, 0))
}

/**
 * Normalize tensor to unit norm
 */
export function normalize(tensor: Tensor): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor
  const n = norm(dense)
  if (n === 0)
    return clone(dense)
  return scale(dense, 1 / n)
}

/**
 * Concatenate tensors along a new or existing index
 */
export function concat(
  tensors: Tensor[],
  indexName: IndexName,
): DenseTensor<number> {
  if (tensors.length === 0) {
    throw new Error('Cannot concatenate empty array')
  }

  const denseTensors = tensors.map(t => t.type === 'sparse' ? toDense(t) : t)

  // Check that all tensors have same shape
  const firstShape = denseTensors[0].shape
  for (let i = 1; i < denseTensors.length; i++) {
    if (denseTensors[i].shape.size !== firstShape.size) {
      throw new Error('All tensors must have same size for concatenation')
    }
  }

  // Create new shape with additional index
  const newIndices: TensorIndex[] = [
    { name: indexName, size: tensors.length },
    ...firstShape.indices,
  ]
  const newShape = createShape(newIndices.map(i => ({ name: i.name, size: i.size })))

  // Concatenate data
  const newData: number[] = []
  for (const tensor of denseTensors) {
    newData.push(...tensor.data)
  }

  return createDenseTensor(newShape, newData)
}

/**
 * Slice a tensor along an index
 */
export function slice(
  tensor: Tensor,
  indexName: IndexName,
  start: number,
  end: number,
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  const dimIdx = dense.shape.indices.findIndex(i => i.name === indexName)
  if (dimIdx < 0) {
    throw new Error(`Index ${indexName} not found`)
  }

  const newSize = end - start
  const newIndices = dense.shape.indices.map((idx, i) =>
    i === dimIdx ? { ...idx, size: newSize } : idx,
  )
  const newShape = createShape(newIndices.map(i => ({ name: i.name, size: i.size })))
  const newData: number[] = []

  for (let i = 0; i < dense.data.length; i++) {
    const coords = flatToCoords(i, dense.shape)
    if (coords[dimIdx] >= start && coords[dimIdx] < end) {
      newData.push(dense.data[i])
    }
  }

  return createDenseTensor(newShape, newData)
}

/**
 * Max reduction along an index (like projection but with max instead of sum)
 */
export function maxReduce(
  tensor: Tensor,
  targetIndices: IndexName[],
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  const keepIndices = dense.shape.indices.filter(idx => targetIndices.includes(idx.name))

  if (keepIndices.length === 0) {
    return createDenseTensor(createShape([]), [max(dense)])
  }

  const outShape = createShape(keepIndices.map(idx => ({ name: idx.name, size: idx.size })))
  const outData = new Array(outShape.size).fill(Number.NEGATIVE_INFINITY)

  for (let i = 0; i < dense.data.length; i++) {
    const coords = flatToCoords(i, dense.shape)
    const outCoords = keepIndices.map((idx) => {
      const srcIdx = dense.shape.indices.findIndex(si => si.name === idx.name)
      return coords[srcIdx]
    })
    const outFlatIdx = coordsToFlat(outCoords, outShape)
    outData[outFlatIdx] = Math.max(outData[outFlatIdx], dense.data[i])
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Average reduction along an index
 */
export function avgReduce(
  tensor: Tensor,
  targetIndices: IndexName[],
): DenseTensor<number> {
  const dense = tensor.type === 'sparse' ? toDense(tensor) : tensor

  const keepIndices = dense.shape.indices.filter(idx => targetIndices.includes(idx.name))
  const sumIndices = dense.shape.indices.filter(idx => !targetIndices.includes(idx.name))
  const sumSize = sumIndices.reduce((acc, idx) => acc * idx.size, 1)

  const projected = project(dense, targetIndices)
  return scale(projected, 1 / sumSize)
}

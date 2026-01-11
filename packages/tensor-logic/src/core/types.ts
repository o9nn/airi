/**
 * Tensor Logic Core Types
 *
 * Implements the fundamental types for tensor logic programming:
 * - TensorShape: Defines tensor dimensions
 * - TensorIndex: Named indices for Einstein summation
 * - Tensor: N-dimensional array with named indices
 * - SparseTensor: Efficient representation for Boolean/sparse data
 */

/**
 * A named index for tensor dimensions, enabling Einstein summation notation.
 * Examples: 'i', 'j', 'x', 'y', 't', 'd'
 */
export type IndexName = string

/**
 * Index specification with optional modifiers
 */
export interface TensorIndex {
  /** The index name (e.g., 'i', 'x', 't') */
  name: IndexName
  /** Size of this dimension */
  size: number
  /** Whether this is a virtual index (prefixed with * in notation) */
  virtual?: boolean
  /** Index function offset (e.g., t+1) */
  offset?: number
  /** Division factor for pooling (e.g., x/S) */
  stride?: number
  /** Slice range [start, end) */
  slice?: [number, number]
}

/**
 * Shape specification for a tensor
 */
export interface TensorShape {
  /** Ordered list of indices defining tensor dimensions */
  indices: TensorIndex[]
  /** Total number of elements */
  size: number
}

/**
 * Creates a tensor shape from index specifications
 */
export function createShape(indices: Array<TensorIndex | { name: string, size: number }>): TensorShape {
  const normalizedIndices: TensorIndex[] = indices.map(idx => ({
    name: idx.name,
    size: idx.size,
    virtual: (idx as TensorIndex).virtual ?? false,
    offset: (idx as TensorIndex).offset,
    stride: (idx as TensorIndex).stride,
    slice: (idx as TensorIndex).slice,
  }))

  const size = normalizedIndices.reduce((acc, idx) => acc * (idx.virtual ? 1 : idx.size), 1)

  return { indices: normalizedIndices, size }
}

/**
 * Data type for tensor elements
 */
export type TensorDType = 'float32' | 'float64' | 'int32' | 'bool'

/**
 * Dense tensor representation
 */
export interface DenseTensor<T extends number | boolean = number> {
  type: 'dense'
  /** Tensor shape specification */
  shape: TensorShape
  /** Flat array of tensor data */
  data: T[]
  /** Data type */
  dtype: TensorDType
  /** Get index names */
  indexNames: IndexName[]
}

/**
 * Coordinate format for sparse tensor entries
 */
export interface SparseEntry<T = number> {
  /** Coordinates for each dimension */
  coords: number[]
  /** Value at this coordinate */
  value: T
}

/**
 * Sparse tensor representation (efficient for Boolean/relation tensors)
 */
export interface SparseTensor<T extends number | boolean = boolean> {
  type: 'sparse'
  /** Tensor shape specification */
  shape: TensorShape
  /** Non-zero entries in coordinate format */
  entries: SparseEntry<T>[]
  /** Data type */
  dtype: TensorDType
  /** Default value for unspecified entries */
  defaultValue: T
  /** Get index names */
  indexNames: IndexName[]
}

/**
 * Union type for all tensor representations
 */
export type Tensor<T extends number | boolean = number> = DenseTensor<T> | SparseTensor<T>

/**
 * Creates a new dense tensor with given shape and optional initial data
 */
export function createDenseTensor(
  shape: TensorShape,
  data?: number[],
  dtype: TensorDType = 'float32',
): DenseTensor<number> {
  const tensorData = data ?? new Array(shape.size).fill(0)

  if (tensorData.length !== shape.size) {
    throw new Error(`Data length ${tensorData.length} does not match shape size ${shape.size}`)
  }

  return {
    type: 'dense',
    shape,
    data: tensorData,
    dtype,
    indexNames: shape.indices.map(idx => idx.name),
  }
}

/**
 * Creates a new sparse tensor (efficient for Boolean relations)
 */
export function createSparseTensor<T extends number | boolean = boolean>(
  shape: TensorShape,
  entries: SparseEntry<T>[] = [],
  defaultValue: T = false as T,
  dtype: TensorDType = 'bool',
): SparseTensor<T> {
  return {
    type: 'sparse',
    shape,
    entries,
    dtype,
    defaultValue,
    indexNames: shape.indices.map(idx => idx.name),
  }
}

/**
 * Creates a Boolean relation tensor from tuples
 * Example: createRelation(['x', 'y'], [[0, 1], [1, 2], [2, 0]], [10, 10])
 */
export function createRelation(
  indexNames: IndexName[],
  tuples: number[][],
  sizes: number[],
): SparseTensor<boolean> {
  if (indexNames.length !== sizes.length) {
    throw new Error('Index names and sizes must have same length')
  }

  const shape = createShape(indexNames.map((name, i) => ({ name, size: sizes[i] })))

  const entries: SparseEntry<boolean>[] = tuples.map(tuple => ({
    coords: tuple,
    value: true,
  }))

  return createSparseTensor(shape, entries, false, 'bool')
}

/**
 * Creates an embedding tensor for objects
 * Each row is a random unit vector
 */
export function createEmbedding(
  objectIndexName: IndexName,
  numObjects: number,
  embeddingIndexName: IndexName,
  embeddingDim: number,
): DenseTensor<number> {
  const shape = createShape([
    { name: objectIndexName, size: numObjects },
    { name: embeddingIndexName, size: embeddingDim },
  ])

  const data: number[] = []

  for (let i = 0; i < numObjects; i++) {
    // Generate random vector
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
 * Creates a zeros tensor
 */
export function zeros(shape: TensorShape): DenseTensor<number> {
  return createDenseTensor(shape, new Array(shape.size).fill(0))
}

/**
 * Creates a ones tensor
 */
export function ones(shape: TensorShape): DenseTensor<number> {
  return createDenseTensor(shape, new Array(shape.size).fill(1))
}

/**
 * Creates a tensor filled with a constant value
 */
export function full(shape: TensorShape, value: number): DenseTensor<number> {
  return createDenseTensor(shape, new Array(shape.size).fill(value))
}

/**
 * Creates a random tensor with values in [0, 1)
 */
export function random(shape: TensorShape): DenseTensor<number> {
  return createDenseTensor(shape, Array.from({ length: shape.size }, () => Math.random()))
}

/**
 * Creates a random tensor with normal distribution
 */
export function randn(shape: TensorShape, mean = 0, std = 1): DenseTensor<number> {
  const data: number[] = []
  for (let i = 0; i < shape.size; i++) {
    // Box-Muller transform
    const u1 = Math.random()
    const u2 = Math.random()
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    data.push(z * std + mean)
  }
  return createDenseTensor(shape, data)
}

/**
 * Creates an identity matrix
 */
export function eye(indexName1: IndexName, indexName2: IndexName, size: number): DenseTensor<number> {
  const shape = createShape([
    { name: indexName1, size },
    { name: indexName2, size },
  ])

  const data = new Array(size * size).fill(0)
  for (let i = 0; i < size; i++) {
    data[i * size + i] = 1
  }

  return createDenseTensor(shape, data)
}

/**
 * Gets the rank (number of dimensions) of a tensor
 */
export function rank(tensor: Tensor): number {
  return tensor.shape.indices.length
}

/**
 * Gets the dimensions of a tensor
 */
export function dims(tensor: Tensor): number[] {
  return tensor.shape.indices.map(idx => idx.size)
}

/**
 * Converts a sparse tensor to dense
 */
export function toDense(tensor: SparseTensor): DenseTensor<number> {
  const data = new Array(tensor.shape.size).fill(tensor.defaultValue ? 1 : 0)

  for (const entry of tensor.entries) {
    const flatIdx = coordsToFlat(entry.coords, tensor.shape)
    data[flatIdx] = entry.value ? 1 : 0
  }

  return {
    type: 'dense',
    shape: tensor.shape,
    data,
    dtype: 'float32',
    indexNames: tensor.indexNames,
  }
}

/**
 * Converts a dense tensor to sparse (keeping non-zero entries)
 */
export function toSparse(tensor: DenseTensor, threshold = 0): SparseTensor<boolean> {
  const entries: SparseEntry<boolean>[] = []

  for (let i = 0; i < tensor.data.length; i++) {
    if (Math.abs(tensor.data[i]) > threshold) {
      entries.push({
        coords: flatToCoords(i, tensor.shape),
        value: true,
      })
    }
  }

  return {
    type: 'sparse',
    shape: tensor.shape,
    entries,
    dtype: 'bool',
    defaultValue: false,
    indexNames: tensor.indexNames,
  }
}

/**
 * Converts flat index to multi-dimensional coordinates
 */
export function flatToCoords(flatIdx: number, shape: TensorShape): number[] {
  const coords: number[] = []
  let remaining = flatIdx

  for (let i = shape.indices.length - 1; i >= 0; i--) {
    const size = shape.indices[i].size
    coords.unshift(remaining % size)
    remaining = Math.floor(remaining / size)
  }

  return coords
}

/**
 * Converts multi-dimensional coordinates to flat index
 */
export function coordsToFlat(coords: number[], shape: TensorShape): number {
  let flatIdx = 0
  let multiplier = 1

  for (let i = shape.indices.length - 1; i >= 0; i--) {
    flatIdx += coords[i] * multiplier
    multiplier *= shape.indices[i].size
  }

  return flatIdx
}

/**
 * Gets element at given coordinates
 */
export function getElement(tensor: Tensor, coords: number[]): number | boolean {
  if (tensor.type === 'dense') {
    return tensor.data[coordsToFlat(coords, tensor.shape)]
  }
  else {
    // Sparse tensor lookup
    for (const entry of tensor.entries) {
      if (entry.coords.every((c, i) => c === coords[i])) {
        return entry.value
      }
    }
    return tensor.defaultValue
  }
}

/**
 * Sets element at given coordinates (mutates tensor)
 */
export function setElement(tensor: Tensor, coords: number[], value: number | boolean): void {
  if (tensor.type === 'dense') {
    (tensor as DenseTensor).data[coordsToFlat(coords, tensor.shape)] = value as number
  }
  else {
    const sparse = tensor as SparseTensor
    // Check if entry exists
    const existingIdx = sparse.entries.findIndex(e => e.coords.every((c, i) => c === coords[i]))
    if (existingIdx >= 0) {
      sparse.entries[existingIdx].value = value as boolean
    }
    else if (value !== sparse.defaultValue) {
      sparse.entries.push({ coords: [...coords], value: value as boolean })
    }
  }
}

/**
 * Clones a tensor
 */
export function clone<T extends Tensor>(tensor: T): T {
  if (tensor.type === 'dense') {
    return {
      ...tensor,
      data: [...tensor.data],
      shape: {
        ...tensor.shape,
        indices: tensor.shape.indices.map(idx => ({ ...idx })),
      },
    } as T
  }
  else {
    return {
      ...tensor,
      entries: tensor.entries.map(e => ({ coords: [...e.coords], value: e.value })),
      shape: {
        ...tensor.shape,
        indices: tensor.shape.indices.map(idx => ({ ...idx })),
      },
    } as T
  }
}

/**
 * Reshapes a tensor (must preserve total size)
 */
export function reshape(tensor: DenseTensor, newShape: TensorShape): DenseTensor {
  if (tensor.shape.size !== newShape.size) {
    throw new Error(`Cannot reshape tensor of size ${tensor.shape.size} to size ${newShape.size}`)
  }

  return {
    ...tensor,
    shape: newShape,
    indexNames: newShape.indices.map(idx => idx.name),
  }
}

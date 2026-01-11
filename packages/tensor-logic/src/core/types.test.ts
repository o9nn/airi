import { describe, expect, it } from 'vitest'

import {
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

describe('TensorShape', () => {
  it('should create a shape with indices', () => {
    const shape = createShape([
      { name: 'i', size: 3 },
      { name: 'j', size: 4 },
    ])

    expect(shape.indices).toHaveLength(2)
    expect(shape.indices[0].name).toBe('i')
    expect(shape.indices[0].size).toBe(3)
    expect(shape.indices[1].name).toBe('j')
    expect(shape.indices[1].size).toBe(4)
    expect(shape.size).toBe(12)
  })

  it('should handle virtual indices', () => {
    const shape = createShape([
      { name: 'i', size: 3, virtual: true },
      { name: 'j', size: 4 },
    ])

    // Virtual indices don't contribute to size
    expect(shape.size).toBe(4)
  })

  it('should create empty shape (scalar)', () => {
    const shape = createShape([])
    expect(shape.indices).toHaveLength(0)
    expect(shape.size).toBe(1)
  })
})

describe('DenseTensor', () => {
  it('should create a tensor with data', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = createDenseTensor(shape, [1, 2, 3])

    expect(tensor.type).toBe('dense')
    expect(tensor.data).toEqual([1, 2, 3])
    expect(tensor.indexNames).toEqual(['i'])
  })

  it('should create a tensor with zeros if no data provided', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = createDenseTensor(shape)

    expect(tensor.data).toEqual([0, 0, 0])
  })

  it('should throw if data length does not match shape', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    expect(() => createDenseTensor(shape, [1, 2])).toThrow()
  })

  it('should support different dtypes', () => {
    const shape = createShape([{ name: 'i', size: 2 }])
    const tensor = createDenseTensor(shape, [1, 2], 'float64')

    expect(tensor.dtype).toBe('float64')
  })
})

describe('SparseTensor', () => {
  it('should create a sparse tensor', () => {
    const shape = createShape([
      { name: 'x', size: 10 },
      { name: 'y', size: 10 },
    ])
    const tensor = createSparseTensor(shape, [
      { coords: [0, 1], value: true },
      { coords: [2, 3], value: true },
    ])

    expect(tensor.type).toBe('sparse')
    expect(tensor.entries).toHaveLength(2)
    expect(tensor.defaultValue).toBe(false)
  })

  it('should create a relation from tuples', () => {
    const relation = createRelation(['x', 'y'], [[0, 1], [1, 2], [2, 0]], [5, 5])

    expect(relation.type).toBe('sparse')
    expect(relation.entries).toHaveLength(3)
    expect(relation.indexNames).toEqual(['x', 'y'])
  })
})

describe('createEmbedding', () => {
  it('should create normalized embeddings', () => {
    const emb = createEmbedding('x', 5, 'd', 10)

    expect(emb.shape.indices[0].size).toBe(5)
    expect(emb.shape.indices[1].size).toBe(10)
    expect(emb.indexNames).toEqual(['x', 'd'])

    // Check normalization (each row should be unit vector)
    for (let i = 0; i < 5; i++) {
      let norm = 0
      for (let j = 0; j < 10; j++) {
        norm += emb.data[i * 10 + j] ** 2
      }
      expect(Math.abs(Math.sqrt(norm) - 1)).toBeLessThan(1e-6)
    }
  })
})

describe('tensor creation utilities', () => {
  it('should create zeros tensor', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = zeros(shape)

    expect(tensor.data).toEqual([0, 0, 0])
  })

  it('should create ones tensor', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = ones(shape)

    expect(tensor.data).toEqual([1, 1, 1])
  })

  it('should create full tensor', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = full(shape, 5)

    expect(tensor.data).toEqual([5, 5, 5])
  })

  it('should create random tensor', () => {
    const shape = createShape([{ name: 'i', size: 100 }])
    const tensor = random(shape)

    // All values should be in [0, 1)
    for (const v of tensor.data) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })

  it('should create randn tensor', () => {
    const shape = createShape([{ name: 'i', size: 1000 }])
    const tensor = randn(shape, 0, 1)

    // Mean should be close to 0, std close to 1
    const mean = tensor.data.reduce((a, b) => a + b, 0) / tensor.data.length
    const variance = tensor.data.reduce((a, v) => a + (v - mean) ** 2, 0) / tensor.data.length

    expect(Math.abs(mean)).toBeLessThan(0.2)
    expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.2)
  })

  it('should create identity matrix', () => {
    const tensor = eye('i', 'j', 3)

    expect(tensor.data).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ])
  })
})

describe('coordinate conversion', () => {
  it('should convert flat to coords', () => {
    const shape = createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 3 },
    ])

    expect(flatToCoords(0, shape)).toEqual([0, 0])
    expect(flatToCoords(1, shape)).toEqual([0, 1])
    expect(flatToCoords(2, shape)).toEqual([0, 2])
    expect(flatToCoords(3, shape)).toEqual([1, 0])
    expect(flatToCoords(5, shape)).toEqual([1, 2])
  })

  it('should convert coords to flat', () => {
    const shape = createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 3 },
    ])

    expect(coordsToFlat([0, 0], shape)).toBe(0)
    expect(coordsToFlat([0, 1], shape)).toBe(1)
    expect(coordsToFlat([1, 0], shape)).toBe(3)
    expect(coordsToFlat([1, 2], shape)).toBe(5)
  })

  it('should be reversible', () => {
    const shape = createShape([
      { name: 'i', size: 3 },
      { name: 'j', size: 4 },
      { name: 'k', size: 2 },
    ])

    for (let i = 0; i < shape.size; i++) {
      const coords = flatToCoords(i, shape)
      expect(coordsToFlat(coords, shape)).toBe(i)
    }
  })
})

describe('element access', () => {
  it('should get element from dense tensor', () => {
    const shape = createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 2 },
    ])
    const tensor = createDenseTensor(shape, [1, 2, 3, 4])

    expect(getElement(tensor, [0, 0])).toBe(1)
    expect(getElement(tensor, [0, 1])).toBe(2)
    expect(getElement(tensor, [1, 0])).toBe(3)
    expect(getElement(tensor, [1, 1])).toBe(4)
  })

  it('should get element from sparse tensor', () => {
    const relation = createRelation(['x', 'y'], [[0, 1], [2, 3]], [5, 5])

    expect(getElement(relation, [0, 1])).toBe(true)
    expect(getElement(relation, [2, 3])).toBe(true)
    expect(getElement(relation, [0, 0])).toBe(false)
  })

  it('should set element in dense tensor', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = zeros(shape)

    setElement(tensor, [1], 5)
    expect(tensor.data).toEqual([0, 5, 0])
  })

  it('should set element in sparse tensor', () => {
    const shape = createShape([
      { name: 'x', size: 3 },
      { name: 'y', size: 3 },
    ])
    const tensor = createSparseTensor<boolean>(shape, [], false)

    setElement(tensor, [1, 2], true)
    expect(tensor.entries).toHaveLength(1)
    expect(getElement(tensor, [1, 2])).toBe(true)
  })
})

describe('tensor utilities', () => {
  it('should get rank', () => {
    const tensor = zeros(createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 3 },
      { name: 'k', size: 4 },
    ]))

    expect(rank(tensor)).toBe(3)
  })

  it('should get dims', () => {
    const tensor = zeros(createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 3 },
      { name: 'k', size: 4 },
    ]))

    expect(dims(tensor)).toEqual([2, 3, 4])
  })

  it('should clone tensor', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const original = createDenseTensor(shape, [1, 2, 3])
    const cloned = clone(original)

    // Modify original
    original.data[0] = 100

    // Clone should be unchanged
    expect(cloned.data[0]).toBe(1)
  })

  it('should reshape tensor', () => {
    const shape1 = createShape([{ name: 'i', size: 6 }])
    const tensor = createDenseTensor(shape1, [1, 2, 3, 4, 5, 6])

    const shape2 = createShape([
      { name: 'j', size: 2 },
      { name: 'k', size: 3 },
    ])
    const reshaped = reshape(tensor, shape2)

    expect(reshaped.data).toEqual([1, 2, 3, 4, 5, 6])
    expect(reshaped.indexNames).toEqual(['j', 'k'])
  })

  it('should throw if reshape size mismatch', () => {
    const shape1 = createShape([{ name: 'i', size: 6 }])
    const tensor = createDenseTensor(shape1, [1, 2, 3, 4, 5, 6])

    const shape2 = createShape([{ name: 'j', size: 5 }])
    expect(() => reshape(tensor, shape2)).toThrow()
  })
})

describe('sparse/dense conversion', () => {
  it('should convert sparse to dense', () => {
    const relation = createRelation(['x', 'y'], [[0, 1], [1, 0]], [2, 2])
    const dense = toDense(relation)

    expect(dense.type).toBe('dense')
    expect(dense.data).toEqual([0, 1, 1, 0])
  })

  it('should convert dense to sparse', () => {
    const shape = createShape([{ name: 'i', size: 4 }])
    const tensor = createDenseTensor(shape, [0, 1, 0, 2])
    const sparse = toSparse(tensor)

    expect(sparse.type).toBe('sparse')
    expect(sparse.entries).toHaveLength(2)
  })
})

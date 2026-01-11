import { describe, expect, it } from 'vitest'

import {
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
import {
  createDenseTensor,
  createRelation,
  createShape,
} from './types'

describe('project', () => {
  it('should project out indices (sum reduction)', () => {
    const shape = createShape([
      { name: 'i', size: 2 },
      { name: 'j', size: 3 },
    ])
    const tensor = createDenseTensor(shape, [1, 2, 3, 4, 5, 6])

    const result = project(tensor, ['i'])

    expect(result.indexNames).toEqual(['i'])
    expect(result.data).toEqual([6, 15]) // [1+2+3, 4+5+6]
  })

  it('should return same tensor if no projection needed', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = createDenseTensor(shape, [1, 2, 3])

    const result = project(tensor, ['i'])
    expect(result.data).toEqual([1, 2, 3])
  })

  it('should project to scalar', () => {
    const shape = createShape([{ name: 'i', size: 3 }])
    const tensor = createDenseTensor(shape, [1, 2, 3])

    const result = project(tensor, [])
    expect(result.data).toEqual([6])
  })
})

describe('join', () => {
  it('should perform matrix multiplication', () => {
    // A[i,j] * B[j,k] -> C[i,k]
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 2 }]),
      [1, 2, 3, 4],
    )
    const B = createDenseTensor(
      createShape([{ name: 'j', size: 2 }, { name: 'k', size: 2 }]),
      [5, 6, 7, 8],
    )

    const result = join(A, B)

    expect(result.indexNames).toEqual(['i', 'k'])
    expect(result.data).toEqual([
      19, 22, // [1*5+2*7, 1*6+2*8]
      43, 50, // [3*5+4*7, 3*6+4*8]
    ])
  })

  it('should perform dot product', () => {
    // A[i] * B[i] -> scalar
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )
    const B = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [4, 5, 6],
    )

    const result = join(A, B)

    expect(result.data).toEqual([32]) // 1*4 + 2*5 + 3*6
  })

  it('should perform Kronecker product (no shared indices)', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [1, 2],
    )
    const B = createDenseTensor(
      createShape([{ name: 'j', size: 2 }]),
      [3, 4],
    )

    const result = join(A, B)

    expect(result.indexNames).toEqual(['i', 'j'])
    expect(result.data).toEqual([3, 4, 6, 8])
  })

  it('should perform Hadamard product (all indices shared)', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )
    const B = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [4, 5, 6],
    )

    const result = hadamard(A, B)

    expect(result.data).toEqual([4, 10, 18])
  })
})

describe('einsum', () => {
  it('should perform matrix multiply with notation', () => {
    const A = createDenseTensor(
      createShape([{ name: 'a', size: 2 }, { name: 'b', size: 2 }]),
      [1, 2, 3, 4],
    )
    const B = createDenseTensor(
      createShape([{ name: 'c', size: 2 }, { name: 'd', size: 2 }]),
      [5, 6, 7, 8],
    )

    const result = einsum('ij,jk->ik', A, B)

    expect(result.data).toEqual([19, 22, 43, 50])
  })

  it('should perform batch dot product with einsum', () => {
    const A = createDenseTensor(
      createShape([{ name: 'a', size: 2 }, { name: 'b', size: 3 }]),
      [1, 2, 3, 4, 5, 6],
    )
    const B = createDenseTensor(
      createShape([{ name: 'c', size: 3 }]),
      [1, 1, 1],
    )

    const result = einsum('ij,j->i', A, B)

    // Row sums: [1+2+3, 4+5+6] = [6, 15]
    expect(result.data).toEqual([6, 15])
  })

  it('should perform outer product', () => {
    const A = createDenseTensor(
      createShape([{ name: 'a', size: 2 }]),
      [1, 2],
    )
    const B = createDenseTensor(
      createShape([{ name: 'b', size: 2 }]),
      [3, 4],
    )

    const result = einsum('i,j->ij', A, B)

    expect(result.data).toEqual([3, 4, 6, 8])
  })
})

describe('arithmetic operations', () => {
  it('should add tensors', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])
    const B = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [4, 5, 6])

    const result = add(A, B)
    expect(result.data).toEqual([5, 7, 9])
  })

  it('should subtract tensors', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [4, 5, 6])
    const B = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    const result = subtract(A, B)
    expect(result.data).toEqual([3, 3, 3])
  })

  it('should scale tensor', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    const result = scale(A, 2)
    expect(result.data).toEqual([2, 4, 6])
  })

  it('should divide tensors', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [4, 6, 8])
    const B = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [2, 2, 2])

    const result = divide(A, B)
    expect(result.data).toEqual([2, 3, 4])
  })

  it('should handle division by zero', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])
    const B = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [0, 1, 0])

    const result = divide(A, B)
    expect(result.data[1]).toBe(2)
    expect(result.data[0]).toBe(0) // 1/0 -> 0 (safe division)
  })
})

describe('transpose', () => {
  it('should transpose 2D tensor', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 3 }]),
      [1, 2, 3, 4, 5, 6],
    )

    const result = transpose(A)

    expect(result.indexNames).toEqual(['j', 'i'])
    expect(result.shape.indices[0].size).toBe(3)
    expect(result.shape.indices[1].size).toBe(2)
    expect(result.data).toEqual([1, 4, 2, 5, 3, 6])
  })

  it('should transpose specific indices', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 3 }]),
      [1, 2, 3, 4, 5, 6],
    )

    const result = transpose(A, 'j', 'i')
    expect(result.data).toEqual([1, 4, 2, 5, 3, 6])
  })
})

describe('reduction operations', () => {
  it('should compute sum', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 2, 3, 4])
    expect(sum(A)).toBe(10)
  })

  it('should compute mean', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 2, 3, 4])
    expect(mean(A)).toBe(2.5)
  })

  it('should compute max', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 4, 2, 3])
    expect(max(A)).toBe(4)
  })

  it('should compute min', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 4, 2, 3])
    expect(min(A)).toBe(1)
  })

  it('should compute argmax', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 4, 2, 3])
    expect(argmax(A)).toBe(1)
  })

  it('should compute argmin', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [1, 4, 2, 3])
    expect(argmin(A)).toBe(0)
  })

  it('should compute norm', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [3, 4, 0])
    expect(norm(A)).toBe(5)
  })

  it('should normalize tensor', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [3, 4, 0])
    const result = normalize(A)
    expect(result.data[0]).toBeCloseTo(0.6)
    expect(result.data[1]).toBeCloseTo(0.8)
    expect(result.data[2]).toBe(0)
  })
})

describe('maxReduce and avgReduce', () => {
  it('should reduce with max', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 3 }]),
      [1, 5, 3, 4, 2, 6],
    )

    const result = maxReduce(A, ['i'])
    expect(result.data).toEqual([5, 6])
  })

  it('should reduce with avg', () => {
    const A = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 3 }]),
      [1, 2, 3, 4, 5, 6],
    )

    const result = avgReduce(A, ['i'])
    expect(result.data).toEqual([2, 5]) // [avg(1,2,3), avg(4,5,6)]
  })
})

describe('concat and slice', () => {
  it('should concatenate tensors', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])
    const B = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [3, 4])
    const C = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [5, 6])

    const result = concat([A, B, C], 'batch')

    expect(result.indexNames).toContain('batch')
    expect(result.data).toEqual([1, 2, 3, 4, 5, 6])
  })

  it('should slice tensor', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 5 }]), [1, 2, 3, 4, 5])

    const result = slice(A, 'i', 1, 4)

    expect(result.shape.indices[0].size).toBe(3)
    expect(result.data).toEqual([2, 3, 4])
  })
})

describe('outer product', () => {
  it('should compute outer product', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])
    const B = createDenseTensor(createShape([{ name: 'j', size: 3 }]), [3, 4, 5])

    const result = outer(A, B)

    expect(result.indexNames).toEqual(['i', 'j'])
    expect(result.data).toEqual([3, 4, 5, 6, 8, 10])
  })

  it('should throw if indices overlap', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])
    const B = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [3, 4, 5])

    expect(() => outer(A, B)).toThrow()
  })
})

describe('sparse tensor operations', () => {
  it('should join sparse tensors', () => {
    const parent = createRelation(['x', 'y'], [[0, 1], [1, 2]], [3, 3])
    const sibling = createRelation(['x', 'y'], [[0, 0], [2, 2]], [3, 3])

    // Join should work on sparse tensors
    const result = join(parent, sibling)
    expect(result.type).toBe('dense')
  })

  it('should project sparse tensors', () => {
    const relation = createRelation(['x', 'y'], [[0, 1], [0, 2], [1, 2]], [3, 3])

    const result = project(relation, ['x'])
    expect(result.data[0]).toBe(2) // x=0 appears twice
    expect(result.data[1]).toBe(1) // x=1 appears once
  })
})

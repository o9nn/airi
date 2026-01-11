import { describe, expect, it } from 'vitest'

import {
  createDenseTensor,
  createRelation,
  createShape,
} from '../core/types'

import {
  BackwardChainingEngine,
  createBackwardEngine,
  createForwardEngine,
  execute,
  ForwardChainingEngine,
} from './engine'

describe('ForwardChainingEngine', () => {
  it('should execute simple assignment', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    const result = execute('Y = X', new Map([['X', X]]))

    expect(result.tensors.has('Y')).toBe(true)
    const Y = result.tensors.get('Y')!
    expect(Y.type).toBe('dense')
    expect((Y as any).data).toEqual([1, 2, 3])
  })

  it('should execute sigmoid activation', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [0, 0, 0])

    const result = execute('Y = sigmoid(X)', new Map([['X', X]]))

    const Y = result.tensors.get('Y')!
    expect((Y as any).data.every((v: number) => Math.abs(v - 0.5) < 0.01)).toBe(true)
  })

  it('should execute step function', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 4 }]), [-1, -0.1, 0.1, 1])

    const result = execute('Y = step(X)', new Map([['X', X]]))

    const Y = result.tensors.get('Y')!
    expect((Y as any).data).toEqual([0, 0, 1, 1])
  })

  it('should execute matrix multiplication', () => {
    const W = createDenseTensor(
      createShape([{ name: 'i', size: 2 }, { name: 'j', size: 2 }]),
      [1, 0, 0, 1], // Identity matrix
    )
    const X = createDenseTensor(
      createShape([{ name: 'j', size: 2 }]),
      [3, 4],
    )

    const result = execute('Y = W[i,j] * X[j]', new Map([['W', W], ['X', X]]))

    const Y = result.tensors.get('Y')!
    expect((Y as any).data).toEqual([3, 4])
  })

  it('should execute multi-step program', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])
    const W1 = createDenseTensor(
      createShape([{ name: 'j', size: 2 }, { name: 'i', size: 2 }]),
      [1, 1, 1, 1],
    )
    const W2 = createDenseTensor(
      createShape([{ name: 'k', size: 1 }, { name: 'j', size: 2 }]),
      [1, 1],
    )

    const program = `
      H = relu(W1[j,i] * X[i])
      Y = sigmoid(W2[k,j] * H[j])
    `

    const result = execute(program, new Map([
      ['X', X],
      ['W1', W1],
      ['W2', W2],
    ]))

    expect(result.tensors.has('H')).toBe(true)
    expect(result.tensors.has('Y')).toBe(true)
  })

  it('should reach fixpoint for recursive computation', () => {
    // Simple fixpoint: Y = 0.5 * Y + 0.5 * X, starting from Y = 0
    // Should converge to Y = X
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [2, 4])
    const Y = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [0, 0])
    const half = createDenseTensor(createShape([]), [0.5])

    const engine = createForwardEngine(
      'Y = half * Y + half * X',
      new Map([['X', X], ['Y', Y], ['half', half]]),
      { maxIterations: 100, convergenceThreshold: 1e-6 },
    )

    const result = engine.execute()

    expect(result.converged).toBe(true)
    const finalY = result.tensors.get('Y')!
    expect(Math.abs((finalY as any).data[0] - 2)).toBeLessThan(0.01)
    expect(Math.abs((finalY as any).data[1] - 4)).toBeLessThan(0.01)
  })

  it('should track history when enabled', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    const result = execute(
      'Y = sigmoid(X)',
      new Map([['X', X]]),
      { trackHistory: true },
    )

    expect(result.history).toBeDefined()
    expect(result.history!.length).toBeGreaterThan(0)
  })

  it('should execute relu activation', () => {
    const X = createDenseTensor(
      createShape([{ name: 'i', size: 4 }]),
      [-2, -1, 1, 2],
    )

    const result = execute('Y = relu(X)', new Map([['X', X]]))

    const Y = result.tensors.get('Y')!
    expect((Y as any).data).toEqual([0, 0, 1, 2])
  })

  it('should execute tanh activation', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [0, 0, 0])

    const result = execute('Y = tanh(X)', new Map([['X', X]]))

    const Y = result.tensors.get('Y')!
    expect((Y as any).data.every((v: number) => v === 0)).toBe(true)
  })

  it('should handle softmax', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    const result = execute('Y = softmax(X)', new Map([['X', X]]))

    const Y = result.tensors.get('Y')!
    const sum = (Y as any).data.reduce((a: number, b: number) => a + b, 0)
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6)
  })
})

describe('BackwardChainingEngine', () => {
  it('should query defined tensor', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    const engine = createBackwardEngine('Y = X', new Map([['X', X]]))
    const Y = engine.query('Y')

    expect(Y).not.toBeNull()
    expect((Y as any).data).toEqual([1, 2, 3])
  })

  it('should return null for undefined tensor', () => {
    const engine = createBackwardEngine('Y = X')
    const result = engine.query('Y')

    expect(result).toBeNull()
  })

  it('should recursively resolve dependencies', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    const program = `
      A = X
      B = sigmoid(A)
      C = relu(B)
    `

    const engine = createBackwardEngine(program, new Map([['X', X]]))
    const C = engine.query('C')

    expect(C).not.toBeNull()
    // sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881, both > 0 so relu passes through
    expect((C as any).data[0]).toBeGreaterThan(0.7)
    expect((C as any).data[1]).toBeGreaterThan(0.8)
  })

  it('should handle circular dependencies gracefully', () => {
    // This creates a cycle: A depends on B, B depends on A
    const program = `
      A = B
      B = A
    `

    const engine = createBackwardEngine(program)

    // Should not hang, should return null
    const result = engine.query('A')
    expect(result).toBeNull()
  })

  it('should cache computed tensors', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    const engine = createBackwardEngine('Y = sigmoid(X)', new Map([['X', X]]))

    // Query twice
    const Y1 = engine.query('Y')
    const Y2 = engine.query('Y')

    // Should return same result
    expect(Y1).not.toBeNull()
    expect(Y2).not.toBeNull()
    expect((Y1 as any).data).toEqual((Y2 as any).data)
  })
})

describe('Datalog-style reasoning', () => {
  it('should execute transitive closure', () => {
    // Parent relation: 0->1, 1->2
    const Parent = createRelation(['x', 'y'], [[0, 1], [1, 2]], [3, 3])

    // Ancestor(x,z) <- Parent(x,y), Parent(y,z)
    // This should derive: 0->2 (grandparent)

    const engine = createForwardEngine(
      'Ancestor = step(Parent[x,y] * Parent[y,z])',
      new Map([['Parent', Parent]]),
      { maxIterations: 10 },
    )

    const result = engine.execute()
    const Ancestor = result.tensors.get('Ancestor')

    expect(Ancestor).toBeDefined()
    // Should have 0->2 connection
  })

  it('should handle sibling relation', () => {
    // Parent: 0->1, 0->2 (siblings 1 and 2)
    const Parent = createRelation(['x', 'y'], [[0, 1], [0, 2]], [3, 3])

    // Sibling(x,y) <- Parent(p,x), Parent(p,y), x != y
    const engine = createForwardEngine(
      'Sibling = step(Parent[p,x] * Parent[p,y])',
      new Map([['Parent', Parent]]),
    )

    const result = engine.execute()
    expect(result.tensors.has('Sibling')).toBe(true)
  })
})

describe('Engine with different nonlinearities', () => {
  const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

  it('should handle leaky_relu', () => {
    const result = execute('Y = leaky_relu(X)', new Map([['X', X]]))
    expect(result.tensors.has('Y')).toBe(true)
  })

  it('should handle elu', () => {
    const result = execute('Y = elu(X)', new Map([['X', X]]))
    expect(result.tensors.has('Y')).toBe(true)
  })

  it('should handle gelu', () => {
    const result = execute('Y = gelu(X)', new Map([['X', X]]))
    expect(result.tensors.has('Y')).toBe(true)
  })

  it('should handle swish', () => {
    const result = execute('Y = swish(X)', new Map([['X', X]]))
    expect(result.tensors.has('Y')).toBe(true)
  })

  it('should handle exp', () => {
    const result = execute('Y = exp(X)', new Map([['X', X]]))
    const Y = result.tensors.get('Y')!
    expect((Y as any).data[0]).toBeCloseTo(Math.E, 4)
  })

  it('should handle log', () => {
    const result = execute('Y = log(X)', new Map([['X', X]]))
    const Y = result.tensors.get('Y')!
    expect((Y as any).data[0]).toBeCloseTo(0, 4)
  })

  it('should handle sqrt', () => {
    const A = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 4, 9])
    const result = execute('Y = sqrt(A)', new Map([['A', A]]))
    const Y = result.tensors.get('Y')!
    expect((Y as any).data).toEqual([1, 2, 3])
  })

  it('should handle lnorm', () => {
    const result = execute('Y = lnorm(X)', new Map([['X', X]]))
    expect(result.tensors.has('Y')).toBe(true)
  })
})

describe('Engine step method', () => {
  it('should execute single step', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    const engine = createForwardEngine('Y = sigmoid(X)', new Map([['X', X]]))

    expect(engine.getTensor('Y')).toBeUndefined()

    engine.step()

    expect(engine.getTensor('Y')).toBeDefined()
  })

  it('should update tensors on each step', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [0, 0])
    const one = createDenseTensor(createShape([]), [1])

    const engine = createForwardEngine(
      'X = X + one',
      new Map([['X', X], ['one', one]]),
    )

    engine.step()
    expect((engine.getTensor('X') as any).data).toEqual([1, 1])

    engine.step()
    expect((engine.getTensor('X') as any).data).toEqual([2, 2])
  })
})

describe('setTensor', () => {
  it('should allow setting tensors', () => {
    const X = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    const engine = createForwardEngine('Y = X')

    engine.setTensor('X', X)
    engine.step()

    expect((engine.getTensor('Y') as any).data).toEqual([1, 2])
  })

  it('should update tensors dynamically', () => {
    const X1 = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])
    const X2 = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [3, 4])

    const engine = createForwardEngine('Y = X')

    engine.setTensor('X', X1)
    engine.step()
    expect((engine.getTensor('Y') as any).data).toEqual([1, 2])

    engine.setTensor('X', X2)
    engine.step()
    expect((engine.getTensor('Y') as any).data).toEqual([3, 4])
  })
})

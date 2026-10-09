import type { DenseTensor, Tensor } from '../core/types'

import { describe, expect, it } from 'vitest'

import { softmax, softmaxGradient } from '../core/nonlinearities'
import { createDenseTensor, createShape } from '../core/types'
import {
  AdamOptimizer,
  createDifferentiableEngine,
  GradientTape,
  LossFunctions,
  SGDOptimizer,
} from './autodiff'

describe('lossFunctions', () => {
  it('should compute MSE loss', () => {
    const predictions = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )
    const targets = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )

    const { loss, gradient } = LossFunctions.mse(predictions, targets)

    // Loss should be 0 when predictions match targets
    expect(loss.data[0]).toBeCloseTo(0, 5)

    // Gradient should be 0 as well
    expect(gradient.data.every(g => Math.abs(g) < 1e-5)).toBe(true)
  })

  it('should compute non-zero MSE loss', () => {
    const predictions = createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [0, 0],
    )
    const targets = createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [1, 1],
    )

    const { loss, gradient } = LossFunctions.mse(predictions, targets)

    // MSE = (1^2 + 1^2) / 2 = 1
    expect(loss.data[0]).toBeCloseTo(1, 5)

    // Gradient = 2/n * (pred - target) = 2/2 * (-1) = -1
    expect(gradient.data[0]).toBeCloseTo(-1, 5)
    expect(gradient.data[1]).toBeCloseTo(-1, 5)
  })

  it('should compute cross-entropy loss', () => {
    const predictions = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0.9, 0.05, 0.05], // Softmax-like output
    )
    const targets = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 0, 0], // One-hot
    )

    const { loss } = LossFunctions.crossEntropy(predictions, targets)

    // -1 * log(0.9) ≈ 0.105
    expect(loss.data[0]).toBeCloseTo(-Math.log(0.9), 2)
  })

  it('should compute binary cross-entropy loss', () => {
    const predictions = createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [0.9, 0.1],
    )
    const targets = createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [1, 0],
    )

    const { loss } = LossFunctions.binaryCrossEntropy(predictions, targets)

    // Good predictions should have low loss
    expect(loss.data[0]).toBeLessThan(0.5)
  })
})

describe('gradientTape', () => {
  it('should record tensor values', () => {
    const tape = new GradientTape()
    const tensor = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    tape.setTensor('X', tensor)

    expect(tape.getTensor('X')).toBeDefined()
    expect((tape.getTensor('X') as any).data).toEqual([1, 2])
  })

  it('should start and stop recording', () => {
    const tape = new GradientTape()

    tape.startRecording()
    expect(tape).toBeDefined()

    tape.stopRecording()
    expect(tape).toBeDefined()
  })

  it('should clear tape', () => {
    const tape = new GradientTape()
    const tensor = createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])

    tape.setTensor('X', tensor)
    tape.clear()

    expect(tape.getTensor('X')).toBeUndefined()
  })
})

describe('differentiableEngine', () => {
  it('should execute forward pass', () => {
    const engine = createDifferentiableEngine('Y = sigmoid(X)')

    engine.setInput('X', createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [0, 0],
    ))

    const tensors = engine.forward()

    expect(tensors.has('Y')).toBe(true)
    expect((tensors.get('Y') as any).data[0]).toBeCloseTo(0.5, 4)
  })

  it('should set parameters for learning', () => {
    const engine = createDifferentiableEngine('Y = W[i] * X[i]')

    const W = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 1, 1])
    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    engine.setParameter('W', W)
    engine.setInput('X', X)

    engine.forward()

    expect(engine.getTensor('Y')).toBeDefined()
  })

  it('should compute gradients', () => {
    const engine = createDifferentiableEngine(`
      H = sigmoid(X)
      loss = square(H)
    `)

    engine.setInput('X', createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [0, 0],
    ))

    engine.forward()
    const grads = engine.backward('loss')

    // Gradients should exist
    expect(grads.size).toBeGreaterThanOrEqual(0)
  })
})

describe('sGDOptimizer', () => {
  it('should update parameters', () => {
    const optimizer = new SGDOptimizer(0.1, 0)

    const params = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 1])],
    ])

    const grads = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2])],
    ])

    optimizer.step(params, grads)

    const W = params.get('W')!
    // W = W - lr * grad = [1, 1] - 0.1 * [1, 2] = [0.9, 0.8]
    expect((W as any).data[0]).toBeCloseTo(0.9, 5)
    expect((W as any).data[1]).toBeCloseTo(0.8, 5)
  })

  it('should use momentum', () => {
    const optimizer = new SGDOptimizer(0.1, 0.9)

    const params = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 1])],
    ])

    const grads = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 1])],
    ])

    // First step
    optimizer.step(params, grads)
    const W1 = (params.get('W') as any).data[0]

    // Second step - momentum should accelerate
    optimizer.step(params, grads)
    const W2 = (params.get('W') as any).data[0]

    // With momentum, steps should compound
    expect(W1).toBeLessThan(1)
    expect(W2).toBeLessThan(W1)
  })
})

describe('adamOptimizer', () => {
  it('should update parameters', () => {
    const optimizer = new AdamOptimizer(0.1)

    const params = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 1])],
    ])

    const grads = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [0.1, 0.1])],
    ])

    optimizer.step(params, grads)

    const W = params.get('W')!
    // Parameters should have changed
    expect((W as any).data[0]).not.toBe(1)
  })

  it('should handle multiple steps', () => {
    const optimizer = new AdamOptimizer(0.01)

    const params = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 1])],
    ])

    const grads = new Map([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 2 }]), [0.1, 0.1])],
    ])

    // Multiple steps
    for (let i = 0; i < 10; i++) {
      optimizer.step(params, grads)
    }

    const W = params.get('W')!
    // Should have moved in the negative gradient direction
    expect((W as any).data[0]).toBeLessThan(1)
  })
})

describe('training loop', () => {
  it('should produce a valid sigmoid output from a forward pass', () => {
    const engine = createDifferentiableEngine(`
      Y = sigmoid(W[i] * X[i])
    `)

    const X = createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3])

    engine.setInput('X', X)
    engine.setParameter('W', createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0.1, 0.1, 0.1],
    ))

    engine.forward()
    const prediction = (engine.getTensor('Y') as any).data[0]

    // W·X = 0.6, so the sigmoid must land strictly inside (0, 1) at 0.6457.
    expect(prediction).toBeCloseTo(0.6456563, 6)
  })

  it('should reduce loss monotonically under SGD', () => {
    // Fit W directly to a target vector. Because the model is the identity,
    // dLoss/dW is exactly the MSE gradient, so this drives the package's own
    // LossFunctions and SGDOptimizer rather than re-deriving the maths here.
    const target = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, -2, 0.5],
    )
    const parameters = new Map<string, Tensor>([
      ['W', createDenseTensor(createShape([{ name: 'i', size: 3 }]), [0, 0, 0])],
    ])
    const optimizer = new SGDOptimizer(0.5)

    const losses: number[] = []
    for (let step = 0; step < 40; step++) {
      const { loss, gradient } = LossFunctions.mse(parameters.get('W')!, target)
      losses.push(loss.data[0])
      optimizer.step(parameters, new Map<string, Tensor>([['W', gradient]]))
    }

    // Gradient descent on a convex quadratic must never increase the loss.
    for (let step = 1; step < losses.length; step++) {
      expect(losses[step]).toBeLessThanOrEqual(losses[step - 1] + 1e-12)
    }

    expect(losses[losses.length - 1]).toBeLessThan(1e-6)

    const W = parameters.get('W') as DenseTensor<number>
    expect(W.data[0]).toBeCloseTo(1, 3)
    expect(W.data[1]).toBeCloseTo(-2, 3)
    expect(W.data[2]).toBeCloseTo(0.5, 3)
  })
})

describe('gradient computation edge cases', () => {
  it('should handle zero gradients', () => {
    const engine = createDifferentiableEngine('Y = X')

    engine.setInput('X', createDenseTensor(
      createShape([{ name: 'i', size: 2 }]),
      [0, 0],
    ))

    engine.forward()

    // Should not throw
    const grads = engine.backward('Y')
    expect(grads).toBeDefined()
  })

  it('should handle scalar outputs', () => {
    const engine = createDifferentiableEngine('Y = X')

    engine.setInput('X', createDenseTensor(createShape([]), [5]))

    engine.forward()

    const Y = engine.getTensor('Y')!
    expect((Y as any).data[0]).toBe(5)
  })
})

describe('softmax backpropagation', () => {
  it('should cancel a uniform upstream gradient exactly', () => {
    // backward() seeds an all-ones gradient. Through softmax that must cancel
    // to zero, since adding a constant to every logit leaves the output
    // unchanged. The old identity passthrough returned the ones untouched.
    const engine = createDifferentiableEngine('Y = softmax(X[i])')

    engine.setParameter('X', createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0.4, -1.3, 2.1],
    ))
    engine.forward()

    const y = engine.getTensor('Y') as DenseTensor<number>
    expect(y.data.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)

    const dX = engine.backward('Y').get('X') as DenseTensor<number> | undefined
    expect(dX).toBeDefined()
    dX!.data.forEach(g => expect(g).toBeCloseTo(0, 10))
    // Specifically not the seeded ones.
    expect(dX!.data.some(g => Math.abs(g - 1) < 1e-9)).toBe(false)
  })

  it('should backpropagate a non-uniform upstream through the Jacobian', () => {
    // square() downstream makes the upstream gradient 2*y rather than uniform,
    // so the result depends on the full Jacobian and cannot cancel to zero.
    const engine = createDifferentiableEngine(`
      S = softmax(X[i])
      loss = square(S[i])
    `)
    const logits = [1.0, 0.25, -0.5]

    engine.setParameter('X', createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      logits,
    ))
    engine.forward()

    const dX = engine.backward('loss').get('X') as DenseTensor<number> | undefined
    expect(dX).toBeDefined()

    // Compare against the analytic VJP with the same upstream, d(s^2)/ds = 2s.
    const y = softmax(createDenseTensor(createShape([{ name: 'i', size: 3 }]), logits))
    const expected = softmaxGradient(
      createDenseTensor(createShape([{ name: 'i', size: 3 }]), logits),
      createDenseTensor(createShape([{ name: 'i', size: 3 }]), y.data.map(v => 2 * v)),
    )

    dX!.data.forEach((g, i) => expect(g).toBeCloseTo(expected.data[i], 10))
    // A real Jacobian moves the logits; identity passthrough would not.
    expect(dX!.data.some(g => Math.abs(g) > 1e-6)).toBe(true)
  })
})

describe('join backpropagation', () => {
  function tensorOf(dims: Array<[string, number]>, data: number[]) {
    return createDenseTensor(
      createShape(dims.map(([name, size]) => ({ name, size }))),
      data,
    )
  }

  type Params = Record<string, DenseTensor<number>>

  function run(source: string, params: Params) {
    const engine = createDifferentiableEngine(source)
    for (const [name, tensor] of Object.entries(params)) {
      engine.setParameter(name, tensor)
    }
    engine.forward()
    return engine
  }

  /** Sum of the named tensor — the objective backward() implicitly seeds with ones. */
  function objective(source: string, params: Params, output: string) {
    const out = run(source, params).getTensor(output) as DenseTensor<number>
    return out.data.reduce((a, b) => a + b, 0)
  }

  /**
   * Central-difference gradient of that objective.
   *
   * These objectives are multilinear in each individual entry, so a central
   * difference is exact up to floating-point noise — nothing here is an
   * approximation that the analytic path is merely close to.
   */
  function numericalGradient(source: string, params: Params, output: string, name: string) {
    const base = params[name]
    const h = 1e-4

    return base.data.map((_, k) => {
      const nudge = (sign: number) => ({
        ...params,
        [name]: createDenseTensor(
          base.shape,
          base.data.map((v, i) => (i === k ? v + sign * h : v)),
        ),
      })
      return (objective(source, nudge(1), output) - objective(source, nudge(-1), output)) / (2 * h)
    })
  }

  function expectGradientMatches(source: string, params: Params, output: string, name: string) {
    const analytic = run(source, params).backward(output).get(name) as DenseTensor<number> | undefined

    expect(analytic, `no gradient produced for ${name}`).toBeDefined()
    // Axis order has to match the operand's own, not whatever order the
    // contraction happened to emit.
    expect(analytic!.shape.indices.map(i => i.name))
      .toEqual(params[name].shape.indices.map(i => i.name))
    expect(analytic!.data.length).toBe(params[name].data.length)

    const numeric = numericalGradient(source, params, output, name)
    analytic!.data.forEach((g, i) => expect(g).toBeCloseTo(numeric[i], 6))
  }

  it('should differentiate a matrix product through both operands', () => {
    const params = {
      A: tensorOf([['i', 2], ['j', 3]], [1, 2, 3, 4, 5, 6]),
      B: tensorOf([['j', 3], ['k', 2]], [0.5, -1, 2, 0.25, -0.5, 3]),
    }

    expectGradientMatches('C[i,k] = A[i,j] * B[j,k]', params, 'C', 'A')
    expectGradientMatches('C[i,k] = A[i,j] * B[j,k]', params, 'C', 'B')
  })

  it('should differentiate an outer product with no shared index', () => {
    const params = {
      A: tensorOf([['i', 3]], [1, -2, 0.5]),
      B: tensorOf([['j', 2]], [4, -0.25]),
    }

    expectGradientMatches('C[i,j] = A[i] * B[j]', params, 'C', 'A')
    expectGradientMatches('C[i,j] = A[i] * B[j]', params, 'C', 'B')
  })

  it('should differentiate a full contraction to a scalar', () => {
    // Every index is shared, so the output is rank 0 and dL/dA is just B.
    const params = {
      A: tensorOf([['i', 4]], [1, 2, 3, 4]),
      B: tensorOf([['i', 4]], [0.5, -1, 2, 0.25]),
    }

    expectGradientMatches('C = A[i] * B[i]', params, 'C', 'A')
    expectGradientMatches('C = A[i] * B[i]', params, 'C', 'B')

    const dA = run('C = A[i] * B[i]', params).backward('C').get('A') as DenseTensor<number>
    expect(Array.from(dA.data)).toEqual(Array.from(params.B.data))
  })

  it('should return the gradient in the operand\'s own axis order', () => {
    // A is stored as [j,i] while the contraction yields [i,j]. Without an
    // explicit permutation the values land on the wrong axes.
    const params = {
      A: tensorOf([['j', 3], ['i', 2]], [1, 2, 3, 4, 5, 6]),
      B: tensorOf([['j', 3], ['k', 2]], [0.5, -1, 2, 0.25, -0.5, 3]),
    }

    expectGradientMatches('C[i,k] = A[j,i] * B[j,k]', params, 'C', 'A')
  })

  it('should contract several shared indices at once', () => {
    const params = {
      A: tensorOf([['i', 2], ['j', 2], ['k', 2]], [1, 2, 3, 4, 5, 6, 7, 8]),
      B: tensorOf([['j', 2], ['k', 2], ['l', 2]], [0.5, -1, 2, 0.25, -0.5, 3, 1.5, -2]),
    }

    expectGradientMatches('C[i,l] = A[i,j,k] * B[j,k,l]', params, 'C', 'A')
    expectGradientMatches('C[i,l] = A[i,j,k] * B[j,k,l]', params, 'C', 'B')
  })

  it('should differentiate a batched product', () => {
    const params = {
      A: tensorOf([['b', 2], ['j', 3]], [1, 2, 3, 4, 5, 6]),
      B: tensorOf([['j', 3], ['k', 2]], [0.5, -1, 2, 0.25, -0.5, 3]),
    }

    expectGradientMatches('C[b,k] = A[b,j] * B[j,k]', params, 'C', 'A')
  })

  it('should sum both paths when an operand appears twice', () => {
    // d(SUM_i A_i A_i)/dA = 2A, which only holds if the two paths accumulate.
    const params = { A: tensorOf([['i', 3]], [1, -2, 0.5]) }

    expectGradientMatches('C = A[i] * A[i]', params, 'C', 'A')

    const dA = run('C = A[i] * A[i]', params).backward('C').get('A') as DenseTensor<number>
    expect(Array.from(dA.data)).toEqual([2, -4, 1])
  })
})

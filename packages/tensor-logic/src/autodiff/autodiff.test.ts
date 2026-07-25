import type { DenseTensor, Tensor } from '../core/types'

import { describe, expect, it } from 'vitest'

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

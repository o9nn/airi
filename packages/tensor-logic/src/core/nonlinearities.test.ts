import { describe, expect, it } from 'vitest'

import {
  abs,
  batchNorm,
  clamp,
  cos,
  dropout,
  elu,
  exp,
  gelu,
  leakyRelu,
  lnorm,
  log,
  logSoftmax,
  neg,
  pow,
  reciprocal,
  relu,
  selu,
  sigmoid,
  sigmoidTemperature,
  sin,
  smoothStep,
  softmax,
  softmaxGradient,
  sqrt,
  square,
  step,
  swish,
  tanh,
} from './nonlinearities'
import { createDenseTensor, createShape } from './types'

describe('step (Heaviside)', () => {
  it('should apply step function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 5 }]),
      [-2, -0.1, 0, 0.1, 2],
    )

    const result = step(tensor)
    expect(result.data).toEqual([0, 0, 0, 1, 1])
  })

  it('should support custom threshold', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 4 }]),
      [0.3, 0.5, 0.7, 1.0],
    )

    const result = step(tensor, 0.5)
    expect(result.data).toEqual([0, 0, 1, 1])
  })
})

describe('smoothStep', () => {
  it('should apply smooth step function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-10, 0, 10],
    )

    const result = smoothStep(tensor, 1)
    expect(result.data[0]).toBeCloseTo(0, 4)
    expect(result.data[1]).toBeCloseTo(0.5, 4)
    expect(result.data[2]).toBeCloseTo(1, 4)
  })
})

describe('sigmoid', () => {
  it('should apply sigmoid function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-10, 0, 10],
    )

    const result = sigmoid(tensor)
    expect(result.data[0]).toBeCloseTo(0, 4)
    expect(result.data[1]).toBeCloseTo(0.5, 4)
    expect(result.data[2]).toBeCloseTo(1, 4)
  })

  it('should support temperature', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-1, 0, 1],
    )

    // Higher temperature = smoother
    const lowTemp = sigmoidTemperature(tensor, 0.1)
    const highTemp = sigmoidTemperature(tensor, 10)

    // At T=0.1, sigmoid(-1) should be very close to 0
    expect(lowTemp.data[0]).toBeLessThan(0.01)
    // At T=10, sigmoid(-1) should be closer to 0.5
    expect(highTemp.data[0]).toBeGreaterThan(0.4)
  })
})

describe('relu', () => {
  it('should apply ReLU function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 5 }]),
      [-2, -0.5, 0, 0.5, 2],
    )

    const result = relu(tensor)
    expect(result.data).toEqual([0, 0, 0, 0.5, 2])
  })
})

describe('leakyRelu', () => {
  it('should apply leaky ReLU function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-2, 0, 2],
    )

    const result = leakyRelu(tensor, 0.1)
    expect(result.data).toEqual([-0.2, 0, 2])
  })
})

describe('elu', () => {
  it('should apply ELU function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-2, 0, 2],
    )

    const result = elu(tensor)
    expect(result.data[0]).toBeCloseTo(Math.exp(-2) - 1)
    expect(result.data[1]).toBe(0)
    expect(result.data[2]).toBe(2)
  })
})

describe('selu', () => {
  it('should apply SELU function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-1, 0, 1],
    )

    const result = selu(tensor)
    expect(result.data[1]).toBe(0)
    expect(result.data[2]).toBeCloseTo(1.0507, 3)
  })
})

describe('gelu', () => {
  it('should apply GELU function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-1, 0, 1],
    )

    const result = gelu(tensor)
    expect(result.data[0]).toBeCloseTo(-0.159, 2)
    expect(result.data[1]).toBe(0)
    expect(result.data[2]).toBeCloseTo(0.841, 2)
  })
})

describe('swish', () => {
  it('should apply swish function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-1, 0, 1],
    )

    const result = swish(tensor)
    expect(result.data[1]).toBe(0)
    // swish(1) = 1 * sigmoid(1) ≈ 0.731
    expect(result.data[2]).toBeCloseTo(0.731, 2)
  })
})

describe('tanh', () => {
  it('should apply tanh function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-10, 0, 10],
    )

    const result = tanh(tensor)
    expect(result.data[0]).toBeCloseTo(-1, 4)
    expect(result.data[1]).toBe(0)
    expect(result.data[2]).toBeCloseTo(1, 4)
  })
})

describe('softmax', () => {
  it('should apply softmax function', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )

    const result = softmax(tensor)

    // Sum should be 1
    const total = result.data.reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 6)

    // Larger values should have higher probability
    expect(result.data[2]).toBeGreaterThan(result.data[1])
    expect(result.data[1]).toBeGreaterThan(result.data[0])
  })

  it('should handle numerical stability', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1000, 1001, 1002],
    )

    const result = softmax(tensor)

    // Should not overflow
    expect(result.data.every(v => Number.isFinite(v))).toBe(true)
    expect(result.data.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })

  it('should work on 2D tensor with axis', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'batch', size: 2 }, { name: 'class', size: 3 }]),
      [1, 2, 3, 4, 5, 6],
    )

    const result = softmax(tensor, 'class')

    // Each row should sum to 1
    expect(result.data[0] + result.data[1] + result.data[2]).toBeCloseTo(1, 6)
    expect(result.data[3] + result.data[4] + result.data[5]).toBeCloseTo(1, 6)
  })
})

describe('logSoftmax', () => {
  it('should compute log-softmax', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 3],
    )

    const result = logSoftmax(tensor)
    const softmaxResult = softmax(tensor)

    // log-softmax should equal log of softmax
    for (let i = 0; i < 3; i++) {
      expect(result.data[i]).toBeCloseTo(Math.log(softmaxResult.data[i]), 5)
    }
  })
})

describe('lnorm (layer normalization)', () => {
  it('should normalize layer', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 4 }]),
      [1, 2, 3, 4],
    )

    const result = lnorm(tensor)

    // Mean should be close to 0
    const mean = result.data.reduce((a, b) => a + b, 0) / result.data.length
    expect(Math.abs(mean)).toBeLessThan(1e-4)

    // Variance should be close to 1
    const variance = result.data.reduce((a, v) => a + (v - mean) ** 2, 0) / result.data.length
    expect(Math.abs(variance - 1)).toBeLessThan(0.1)
  })
})

describe('batchNorm', () => {
  it('should normalize across batch', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'batch', size: 4 }, { name: 'feature', size: 2 }]),
      [1, 10, 2, 20, 3, 30, 4, 40],
    )

    const result = batchNorm(tensor, 'batch')

    // Each feature should have mean ~0 and variance ~1 across batch
    expect(result.data.every(v => Number.isFinite(v))).toBe(true)
  })
})

describe('dropout', () => {
  it('should zero some elements in training', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 100 }]),
      Array.from({ length: 100 }, () => 1),
    )

    const result = dropout(tensor, 0.5, true)

    // Some should be zero, some should be scaled
    const zeros = result.data.filter(v => v === 0).length
    const nonZeros = result.data.filter(v => v !== 0).length

    expect(zeros).toBeGreaterThan(0)
    expect(nonZeros).toBeGreaterThan(0)

    // Non-zeros should be scaled by 1/(1-p) = 2
    const firstNonZero = result.data.find(v => v !== 0)!
    expect(firstNonZero).toBeCloseTo(2, 5)
  })

  it('should not modify in eval mode', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 10 }]),
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )

    const result = dropout(tensor, 0.5, false)
    expect(result.data).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  })
})

describe('math functions', () => {
  it('should compute abs', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 4 }]),
      [-2, -1, 1, 2],
    )

    const result = abs(tensor)
    expect(result.data).toEqual([2, 1, 1, 2])
  })

  it('should compute square', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [-2, 0, 3],
    )

    const result = square(tensor)
    expect(result.data).toEqual([4, 0, 9])
  })

  it('should compute sqrt', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0, 4, 9],
    )

    const result = sqrt(tensor)
    expect(result.data).toEqual([0, 2, 3])
  })

  it('should compute log', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, Math.E, Math.E ** 2],
    )

    const result = log(tensor)
    expect(result.data[0]).toBeCloseTo(0, 5)
    expect(result.data[1]).toBeCloseTo(1, 5)
    expect(result.data[2]).toBeCloseTo(2, 5)
  })

  it('should compute exp', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0, 1, 2],
    )

    const result = exp(tensor)
    expect(result.data[0]).toBeCloseTo(1, 5)
    expect(result.data[1]).toBeCloseTo(Math.E, 5)
    expect(result.data[2]).toBeCloseTo(Math.E ** 2, 5)
  })

  it('should compute pow', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [2, 3, 4],
    )

    const result = pow(tensor, 2)
    expect(result.data).toEqual([4, 9, 16])
  })

  it('should compute sin', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0, Math.PI / 2, Math.PI],
    )

    const result = sin(tensor)
    expect(result.data[0]).toBeCloseTo(0, 5)
    expect(result.data[1]).toBeCloseTo(1, 5)
    expect(result.data[2]).toBeCloseTo(0, 5)
  })

  it('should compute cos', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [0, Math.PI / 2, Math.PI],
    )

    const result = cos(tensor)
    expect(result.data[0]).toBeCloseTo(1, 5)
    expect(result.data[1]).toBeCloseTo(0, 5)
    expect(result.data[2]).toBeCloseTo(-1, 5)
  })

  it('should compute neg', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, -2, 0],
    )

    const result = neg(tensor)
    // Use toBeCloseTo to handle -0 vs 0 edge case
    expect(result.data[0]).toBeCloseTo(-1, 5)
    expect(result.data[1]).toBeCloseTo(2, 5)
    expect(result.data[2]).toBeCloseTo(0, 5)
  })

  it('should compute reciprocal', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }]),
      [1, 2, 4],
    )

    const result = reciprocal(tensor)
    expect(result.data[0]).toBeCloseTo(1, 5)
    expect(result.data[1]).toBeCloseTo(0.5, 5)
    expect(result.data[2]).toBeCloseTo(0.25, 5)
  })

  it('should clamp values', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 5 }]),
      [-2, 0, 0.5, 1, 2],
    )

    const result = clamp(tensor, 0, 1)
    expect(result.data).toEqual([0, 0, 0.5, 1, 1])
  })
})

describe('softmaxGradient', () => {
  /**
   * Numerical vector-Jacobian product by central differences.
   *
   * Treats L = sum_i upstream_i * softmax(x)_i and differentiates it one input
   * at a time, which is the quantity softmaxGradient claims to return. It
   * shares no code with the analytic path, so agreement is real evidence.
   */
  function numericalGradient(
    data: number[],
    dims: Array<{ name: string, size: number }>,
    upstream: number[],
    axis?: string,
    h = 1e-6,
  ): number[] {
    const objective = (values: number[]) => {
      const y = softmax(createDenseTensor(createShape(dims), values), axis)
      return y.data.reduce((acc, v, i) => acc + upstream[i] * v, 0)
    }

    return data.map((_, i) => {
      const plus = [...data]
      const minus = [...data]
      plus[i] += h
      minus[i] -= h
      return (objective(plus) - objective(minus)) / (2 * h)
    })
  }

  it('should match a numerical gradient on a vector', () => {
    const dims = [{ name: 'i', size: 4 }]
    const data = [0.4, -1.3, 2.1, 0.05]
    const upstream = [1, -2, 0.5, 3]

    const analytic = softmaxGradient(
      createDenseTensor(createShape(dims), data),
      createDenseTensor(createShape(dims), upstream),
    )
    const numeric = numericalGradient(data, dims, upstream)

    analytic.data.forEach((g, i) => expect(g).toBeCloseTo(numeric[i], 7))
  })

  it('should match a numerical gradient per row of a matrix', () => {
    // Rank 2 is where a whole-tensor softmax would diverge from a per-axis one,
    // so this is the case the old implementation could not have satisfied.
    const dims = [{ name: 'b', size: 2 }, { name: 'i', size: 3 }]
    const data = [0.1, 0.9, -0.4, 2.0, -1.0, 0.3]
    const upstream = [1, 0, -1, 0.25, 2, -0.5]

    const analytic = softmaxGradient(
      createDenseTensor(createShape(dims), data),
      createDenseTensor(createShape(dims), upstream),
    )
    const numeric = numericalGradient(data, dims, upstream)

    analytic.data.forEach((g, i) => expect(g).toBeCloseTo(numeric[i], 7))
  })

  it('should differentiate along a named axis', () => {
    const dims = [{ name: 'b', size: 2 }, { name: 'i', size: 3 }]
    const data = [0.1, 0.9, -0.4, 2.0, -1.0, 0.3]
    const upstream = [1, 0, -1, 0.25, 2, -0.5]

    const analytic = softmaxGradient(
      createDenseTensor(createShape(dims), data),
      createDenseTensor(createShape(dims), upstream),
      'b',
    )
    const numeric = numericalGradient(data, dims, upstream, 'b')

    analytic.data.forEach((g, i) => expect(g).toBeCloseTo(numeric[i], 7))
  })

  it('should sum to zero within each normalised group', () => {
    // The Jacobian's columns each sum to zero, so no group can shift the total
    // probability mass. This is the invariant the identity passthrough broke.
    const dims = [{ name: 'b', size: 2 }, { name: 'i', size: 3 }]
    const result = softmaxGradient(
      createDenseTensor(createShape(dims), [0.1, 0.9, -0.4, 2.0, -1.0, 0.3]),
      createDenseTensor(createShape(dims), [1, 0, -1, 0.25, 2, -0.5]),
    )

    expect(result.data[0] + result.data[1] + result.data[2]).toBeCloseTo(0, 10)
    expect(result.data[3] + result.data[4] + result.data[5]).toBeCloseTo(0, 10)
  })

  it('should return zero for a uniform upstream gradient', () => {
    // Adding a constant to every logit leaves softmax unchanged, so a uniform
    // upstream gradient must produce no update.
    const dims = [{ name: 'i', size: 4 }]
    const result = softmaxGradient(
      createDenseTensor(createShape(dims), [0.4, -1.3, 2.1, 0.05]),
      createDenseTensor(createShape(dims), [2, 2, 2, 2]),
    )

    result.data.forEach(g => expect(g).toBeCloseTo(0, 12))
  })

  it('should recover a Jacobian column from a one-hot upstream gradient', () => {
    // With upstream = e_k the result is dy_k/dx_i = y_k(d_ik - y_i).
    const dims = [{ name: 'i', size: 3 }]
    const data = [0.5, -0.2, 1.1]
    const y = softmax(createDenseTensor(createShape(dims), data))

    const result = softmaxGradient(
      createDenseTensor(createShape(dims), data),
      createDenseTensor(createShape(dims), [0, 1, 0]),
    )

    for (let i = 0; i < 3; i++) {
      expect(result.data[i]).toBeCloseTo(y.data[1] * ((i === 1 ? 1 : 0) - y.data[i]), 10)
    }
  })

  it('should reject an upstream gradient of the wrong size', () => {
    expect(() => softmaxGradient(
      createDenseTensor(createShape([{ name: 'i', size: 3 }]), [1, 2, 3]),
      createDenseTensor(createShape([{ name: 'i', size: 2 }]), [1, 2]),
    )).toThrow()
  })
})

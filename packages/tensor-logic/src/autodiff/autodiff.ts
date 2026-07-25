/**
 * Tensor Logic Automatic Differentiation
 *
 * Implements automatic differentiation for tensor logic programs:
 * - Gradient computation for tensor equations
 * - Backpropagation through structure
 * - Support for all nonlinearities
 *
 * Key insight from the paper:
 * For Y[...] = T[...]X1[...]...Xn[...]:
 *   ∂Y/∂T = X1 * X2 * ... * Xn (product of other tensors)
 *   ∂Y/∂Xi = T * X1 * ... * Xi-1 * Xi+1 * ... * Xn
 */

import type { DenseTensor, Tensor } from '../core/types'
import type { BinaryOp, Expression, FunctionCall, Program, TensorEquation } from '../parser/ast'

import { add, join, scale, subtract, transpose } from '../core/operations'
import {
  clone,
  createDenseTensor,
  createShape,
  toDense,
  zeros,
} from '../core/types'
import { parse } from '../parser/parser'

/**
 * Tape entry for automatic differentiation
 */
export interface TapeEntry {
  /** Output tensor name */
  output: string
  /** Input tensor names */
  inputs: string[]
  /** The equation that produced this output */
  equation: TensorEquation
  /** Forward pass result */
  value?: Tensor
  /** Backward function */
  backward?: (grad: Tensor) => Map<string, Tensor>
}

/**
 * Gradient tape for recording operations
 */
export class GradientTape {
  private tape: TapeEntry[] = []
  private tensorValues: Map<string, Tensor> = new Map()
  private gradients: Map<string, Tensor> = new Map()
  private recording = true

  /**
   * Start recording operations
   */
  startRecording(): void {
    this.recording = true
    this.tape = []
  }

  /**
   * Stop recording
   */
  stopRecording(): void {
    this.recording = false
  }

  /**
   * Record an operation
   */
  record(entry: TapeEntry): void {
    if (this.recording) {
      this.tape.push(entry)
    }
    if (entry.value) {
      this.tensorValues.set(entry.output, entry.value)
    }
  }

  /**
   * Set a tensor value (for inputs)
   */
  setTensor(name: string, tensor: Tensor): void {
    this.tensorValues.set(name, tensor)
  }

  /**
   * Get a tensor value
   */
  getTensor(name: string): Tensor | undefined {
    return this.tensorValues.get(name)
  }

  /**
   * Compute gradients via backpropagation
   */
  computeGradients(
    lossName: string,
    targetTensors?: string[],
  ): Map<string, Tensor> {
    // Initialize gradient of loss w.r.t. itself as 1
    const loss = this.tensorValues.get(lossName)
    if (!loss) {
      throw new Error(`Loss tensor '${lossName}' not found`)
    }

    const lossGrad = createDenseTensor(loss.shape, new Array(loss.shape.size).fill(1))
    this.gradients.set(lossName, lossGrad)

    // Backpropagate through tape in reverse order
    for (let i = this.tape.length - 1; i >= 0; i--) {
      const entry = this.tape[i]

      // Get upstream gradient
      const upstreamGrad = this.gradients.get(entry.output)
      if (!upstreamGrad)
        continue

      // Compute gradients for inputs
      if (entry.backward) {
        const inputGrads = entry.backward(upstreamGrad)
        for (const [inputName, grad] of inputGrads) {
          // Accumulate gradients
          const existing = this.gradients.get(inputName)
          if (existing) {
            this.gradients.set(inputName, add(existing, grad))
          }
          else {
            this.gradients.set(inputName, grad)
          }
        }
      }
    }

    // Filter to target tensors if specified
    if (targetTensors) {
      const filtered = new Map<string, Tensor>()
      for (const name of targetTensors) {
        const grad = this.gradients.get(name)
        if (grad) {
          filtered.set(name, grad)
        }
      }
      return filtered
    }

    return this.gradients
  }

  /**
   * Get gradient for a specific tensor
   */
  getGradient(name: string): Tensor | undefined {
    return this.gradients.get(name)
  }

  /**
   * Clear the tape
   */
  clear(): void {
    this.tape = []
    this.tensorValues.clear()
    this.gradients.clear()
  }
}

/**
 * Differentiable tensor logic engine
 *
 * Extends forward chaining with automatic differentiation
 */
export class DifferentiableEngine {
  private program: Program
  private tape: GradientTape
  private tensors: Map<string, Tensor>
  private parameters: Set<string>

  constructor(programOrSource: Program | string) {
    this.program = typeof programOrSource === 'string' ? parse(programOrSource) : programOrSource
    this.tape = new GradientTape()
    this.tensors = new Map()
    this.parameters = new Set()
  }

  /**
   * Set input tensor
   */
  setInput(name: string, tensor: Tensor): void {
    this.tensors.set(name, tensor)
    this.tape.setTensor(name, tensor)
  }

  /**
   * Set parameter (learnable tensor)
   */
  setParameter(name: string, tensor: Tensor): void {
    this.tensors.set(name, tensor)
    this.tape.setTensor(name, tensor)
    this.parameters.add(name)
  }

  /**
   * Get tensor value
   */
  getTensor(name: string): Tensor | undefined {
    return this.tensors.get(name)
  }

  /**
   * Forward pass with gradient tracking
   */
  forward(): Map<string, Tensor> {
    this.tape.startRecording()

    for (const stmt of this.program.statements) {
      if (stmt.type === 'equation') {
        this.evaluateWithGrad(stmt)
      }
    }

    this.tape.stopRecording()
    return this.tensors
  }

  /**
   * Backward pass - compute gradients
   */
  backward(lossName: string): Map<string, Tensor> {
    return this.tape.computeGradients(lossName, [...this.parameters])
  }

  /**
   * Get gradient for a parameter
   */
  getGradient(name: string): Tensor | undefined {
    return this.tape.getGradient(name)
  }

  /**
   * Apply gradients using SGD
   */
  applyGradients(learningRate: number): void {
    const gradients = this.tape.computeGradients('loss', [...this.parameters])

    for (const name of this.parameters) {
      const param = this.tensors.get(name)
      const grad = gradients.get(name)

      if (param && grad) {
        const updated = subtract(param, scale(grad, learningRate))
        this.tensors.set(name, updated)
        this.tape.setTensor(name, updated)
      }
    }
  }

  /**
   * Evaluate equation with gradient tracking
   */
  private evaluateWithGrad(eq: TensorEquation): void {
    const inputs = this.collectInputs(eq.rhs)
    const result = this.evaluateExpression(eq.rhs)

    if (result) {
      this.tensors.set(eq.lhs.name, result)
      this.tape.setTensor(eq.lhs.name, result)

      // Record operation with backward function
      this.tape.record({
        output: eq.lhs.name,
        inputs,
        equation: eq,
        value: result,
        backward: grad => this.computeInputGradients(eq.rhs, grad),
      })
    }
  }

  /**
   * Collect all input tensor names from an expression
   */
  private collectInputs(expr: Expression): string[] {
    const inputs: string[] = []

    const collect = (e: Expression): void => {
      switch (e.type) {
        case 'tensor':
          if (!inputs.includes(e.name)) {
            inputs.push(e.name)
          }
          break
        case 'binary':
          collect(e.left)
          collect(e.right)
          break
        case 'unary':
          collect(e.operand)
          break
        case 'function':
          for (const arg of e.arguments) {
            collect(arg)
          }
          break
      }
    }

    collect(expr)
    return inputs
  }

  /**
   * Evaluate an expression
   */
  private evaluateExpression(expr: Expression): Tensor | null {
    switch (expr.type) {
      case 'number':
        return createDenseTensor(createShape([]), [expr.value])

      case 'tensor':
        return this.tensors.get(expr.name) ?? null

      case 'binary':
        return this.evaluateBinary(expr)

      case 'unary':
        if (expr.operator === '-') {
          const operand = this.evaluateExpression(expr.operand)
          if (!operand)
            return null
          const dense = operand.type === 'sparse' ? toDense(operand) : operand
          return createDenseTensor(dense.shape, dense.data.map(x => -x))
        }
        return null

      case 'function':
        return this.evaluateFunction(expr)

      default:
        return null
    }
  }

  /**
   * Evaluate binary operation
   */
  private evaluateBinary(op: BinaryOp): Tensor | null {
    const left = this.evaluateExpression(op.left)
    const right = this.evaluateExpression(op.right)

    if (!left || !right)
      return null

    switch (op.operator) {
      case '+':
        return add(left, right)
      case '-':
        return subtract(left, right)
      case '*':
        return join(left, right)
      case '/': {
        const denseL = left.type === 'sparse' ? toDense(left) : left
        const denseR = right.type === 'sparse' ? toDense(right) : right
        return createDenseTensor(
          denseL.shape,
          denseL.data.map((v, i) => v / (denseR.data[i] || 1)),
        )
      }
      default:
        return null
    }
  }

  /**
   * Evaluate function with forward computation
   */
  private evaluateFunction(call: FunctionCall): Tensor | null {
    const args = call.arguments.map(arg => this.evaluateExpression(arg))
    if (args.includes(null))
      return null

    const x = args[0]!
    const dense = x.type === 'sparse' ? toDense(x) : x

    switch (call.name.toLowerCase()) {
      case 'step':
      case 'h':
        return createDenseTensor(dense.shape, dense.data.map(v => v > 0 ? 1 : 0))

      case 'sigmoid':
      case 'sig':
        return createDenseTensor(dense.shape, dense.data.map(v => 1 / (1 + Math.exp(-v))))

      case 'relu':
        return createDenseTensor(dense.shape, dense.data.map(v => Math.max(0, v)))

      case 'tanh':
        return createDenseTensor(dense.shape, dense.data.map(v => Math.tanh(v)))

      case 'exp':
        return createDenseTensor(dense.shape, dense.data.map(v => Math.exp(v)))

      case 'log':
        return createDenseTensor(dense.shape, dense.data.map(v => Math.log(Math.max(1e-10, v))))

      case 'softmax': {
        const maxVal = Math.max(...dense.data)
        const expData = dense.data.map(v => Math.exp(v - maxVal))
        const sum = expData.reduce((a, b) => a + b, 0)
        return createDenseTensor(dense.shape, expData.map(v => v / sum))
      }

      case 'square':
      case 'sq':
        return createDenseTensor(dense.shape, dense.data.map(v => v * v))

      case 'sqrt':
        return createDenseTensor(dense.shape, dense.data.map(v => Math.sqrt(Math.max(0, v))))

      default:
        return null
    }
  }

  /**
   * Compute gradients for inputs of an expression
   */
  private computeInputGradients(
    expr: Expression,
    upstreamGrad: Tensor,
  ): Map<string, Tensor> {
    const grads = new Map<string, Tensor>()

    this.computeGradsRecursive(expr, upstreamGrad, grads)

    return grads
  }

  /**
   * Recursively compute gradients
   */
  private computeGradsRecursive(
    expr: Expression,
    upstreamGrad: Tensor,
    grads: Map<string, Tensor>,
  ): void {
    switch (expr.type) {
      case 'tensor': {
        // Gradient passes through directly
        const existing = grads.get(expr.name)
        if (existing) {
          grads.set(expr.name, add(existing, upstreamGrad))
        }
        else {
          grads.set(expr.name, clone(upstreamGrad as DenseTensor))
        }
        break
      }

      case 'binary': {
        const left = this.evaluateExpression(expr.left)
        const right = this.evaluateExpression(expr.right)

        if (!left || !right)
          break

        switch (expr.operator) {
          case '+':
            // ∂(A+B)/∂A = 1, ∂(A+B)/∂B = 1
            this.computeGradsRecursive(expr.left, upstreamGrad, grads)
            this.computeGradsRecursive(expr.right, upstreamGrad, grads)
            break

          case '-': {
            // ∂(A-B)/∂A = 1, ∂(A-B)/∂B = -1
            this.computeGradsRecursive(expr.left, upstreamGrad, grads)
            const negGrad = scale(upstreamGrad, -1)
            this.computeGradsRecursive(expr.right, negGrad, grads)
            break
          }

          case '*': {
            // For join: ∂(AB)/∂A = upstream * B, ∂(AB)/∂B = A * upstream
            // This is the key insight from tensor logic paper
            const leftGrad = this.computeJoinGradient(upstreamGrad, right, left)
            const rightGrad = this.computeJoinGradient(upstreamGrad, left, right)
            this.computeGradsRecursive(expr.left, leftGrad, grads)
            this.computeGradsRecursive(expr.right, rightGrad, grads)
            break
          }
        }
        break
      }

      case 'unary':
        if (expr.operator === '-') {
          const negGrad = scale(upstreamGrad, -1)
          this.computeGradsRecursive(expr.operand, negGrad, grads)
        }
        break

      case 'function': {
        const functionGrad = this.computeFunctionGradient(expr, upstreamGrad)
        if (functionGrad && expr.arguments.length > 0) {
          this.computeGradsRecursive(expr.arguments[0], functionGrad, grads)
        }
        break
      }
    }
  }

  /**
   * Compute gradient for join operation
   *
   * For Y = A[i,j] * B[j,k], we have:
   * ∂L/∂A[i,j] = Σ_k (∂L/∂Y[i,k]) * B[j,k]
   * ∂L/∂B[j,k] = Σ_i (∂L/∂Y[i,k]) * A[i,j]
   */
  private computeJoinGradient(
    upstreamGrad: Tensor,
    otherOperand: Tensor,
    targetOperand: Tensor,
  ): Tensor {
    // Simplified gradient computation for join
    // In full implementation, need to properly handle index alignment
    const denseGrad = upstreamGrad.type === 'sparse' ? toDense(upstreamGrad) : upstreamGrad
    const denseOther = otherOperand.type === 'sparse' ? toDense(otherOperand) : otherOperand
    const denseTarget = targetOperand.type === 'sparse' ? toDense(targetOperand) : targetOperand

    // For simple case: gradient shape matches target
    // Join upstream with transposed other operand
    if (denseGrad.shape.indices.length === 2 && denseOther.shape.indices.length === 2) {
      const transposed = transpose(denseOther)
      return join(denseGrad, transposed)
    }

    // Fallback: return scaled upstream gradient
    return createDenseTensor(denseTarget.shape, denseTarget.data.map((_, i) => {
      const gradVal = i < denseGrad.data.length ? denseGrad.data[i] : 0
      return gradVal
    }))
  }

  /**
   * Compute gradient for nonlinear functions
   */
  private computeFunctionGradient(
    call: FunctionCall,
    upstreamGrad: Tensor,
  ): Tensor | null {
    if (call.arguments.length === 0)
      return null

    const input = this.evaluateExpression(call.arguments[0])
    if (!input)
      return null

    const denseInput = input.type === 'sparse' ? toDense(input) : input
    const denseGrad = upstreamGrad.type === 'sparse' ? toDense(upstreamGrad) : upstreamGrad

    switch (call.name.toLowerCase()) {
      case 'step':
      case 'h':
        // Step function has zero gradient almost everywhere
        // Use smoothed gradient for learning
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) => {
          // Use sigmoid derivative as approximation
          const s = 1 / (1 + Math.exp(-10 * v))
          return denseGrad.data[i] * s * (1 - s) * 10
        }))

      case 'sigmoid':
      case 'sig':
        // σ'(x) = σ(x)(1 - σ(x))
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) => {
          const s = 1 / (1 + Math.exp(-v))
          return denseGrad.data[i] * s * (1 - s)
        }))

      case 'relu':
        // ReLU'(x) = 1 if x > 0, else 0
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) =>
          denseGrad.data[i] * (v > 0 ? 1 : 0),
        ))

      case 'tanh':
        // tanh'(x) = 1 - tanh²(x)
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) => {
          const t = Math.tanh(v)
          return denseGrad.data[i] * (1 - t * t)
        }))

      case 'exp':
        // exp'(x) = exp(x)
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) =>
          denseGrad.data[i] * Math.exp(v),
        ))

      case 'log':
        // log'(x) = 1/x
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) =>
          denseGrad.data[i] / Math.max(1e-10, v),
        ))

      case 'square':
      case 'sq':
        // (x²)' = 2x
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) =>
          denseGrad.data[i] * 2 * v,
        ))

      case 'sqrt':
        // sqrt'(x) = 1/(2√x)
        return createDenseTensor(denseInput.shape, denseInput.data.map((v, i) =>
          denseGrad.data[i] / (2 * Math.sqrt(Math.max(1e-10, v))),
        ))

      case 'softmax':
        // Softmax Jacobian is complex, simplified here
        // Full implementation would compute proper Jacobian
        return createDenseTensor(denseInput.shape, denseInput.data.map((_, i) =>
          denseGrad.data[i] ?? 0,
        ))

      default:
        return null
    }
  }
}

/**
 * Loss functions for training
 */
export const LossFunctions = {
  /**
   * Mean Squared Error: (1/n) Σ (y - ŷ)²
   */
  mse(predictions: Tensor, targets: Tensor): { loss: DenseTensor<number>, gradient: DenseTensor<number> } {
    const predDense = predictions.type === 'sparse' ? toDense(predictions) : predictions
    const targetDense = targets.type === 'sparse' ? toDense(targets) : targets

    const n = predDense.data.length
    let lossSum = 0
    const gradData: number[] = []

    for (let i = 0; i < n; i++) {
      const diff = predDense.data[i] - targetDense.data[i]
      lossSum += diff * diff
      gradData.push((2 / n) * diff)
    }

    return {
      loss: createDenseTensor(createShape([]), [lossSum / n]),
      gradient: createDenseTensor(predDense.shape, gradData),
    }
  },

  /**
   * Cross-Entropy Loss: -Σ y log(ŷ)
   */
  crossEntropy(predictions: Tensor, targets: Tensor): { loss: DenseTensor<number>, gradient: DenseTensor<number> } {
    const predDense = predictions.type === 'sparse' ? toDense(predictions) : predictions
    const targetDense = targets.type === 'sparse' ? toDense(targets) : targets

    let lossSum = 0
    const gradData: number[] = []
    const epsilon = 1e-10

    for (let i = 0; i < predDense.data.length; i++) {
      const p = Math.max(epsilon, Math.min(1 - epsilon, predDense.data[i]))
      const t = targetDense.data[i]
      lossSum -= t * Math.log(p)
      gradData.push(-t / p)
    }

    return {
      loss: createDenseTensor(createShape([]), [lossSum]),
      gradient: createDenseTensor(predDense.shape, gradData),
    }
  },

  /**
   * Binary Cross-Entropy
   */
  binaryCrossEntropy(predictions: Tensor, targets: Tensor): { loss: DenseTensor<number>, gradient: DenseTensor<number> } {
    const predDense = predictions.type === 'sparse' ? toDense(predictions) : predictions
    const targetDense = targets.type === 'sparse' ? toDense(targets) : targets

    let lossSum = 0
    const gradData: number[] = []
    const epsilon = 1e-10

    for (let i = 0; i < predDense.data.length; i++) {
      const p = Math.max(epsilon, Math.min(1 - epsilon, predDense.data[i]))
      const t = targetDense.data[i]
      lossSum -= t * Math.log(p) + (1 - t) * Math.log(1 - p)
      gradData.push((p - t) / (p * (1 - p)))
    }

    const n = predDense.data.length
    return {
      loss: createDenseTensor(createShape([]), [lossSum / n]),
      gradient: createDenseTensor(predDense.shape, gradData.map(g => g / n)),
    }
  },
}

/**
 * Optimizers for training
 */
export class SGDOptimizer {
  private learningRate: number
  private momentum: number
  private velocities: Map<string, DenseTensor>

  constructor(learningRate = 0.01, momentum = 0) {
    this.learningRate = learningRate
    this.momentum = momentum
    this.velocities = new Map()
  }

  step(parameters: Map<string, Tensor>, gradients: Map<string, Tensor>): void {
    for (const [name, param] of parameters) {
      const grad = gradients.get(name)
      if (!grad)
        continue

      const paramDense = param.type === 'sparse' ? toDense(param) : param
      const gradDense = grad.type === 'sparse' ? toDense(grad) : grad

      let velocity = this.velocities.get(name)
      if (!velocity) {
        velocity = zeros(paramDense.shape)
        this.velocities.set(name, velocity)
      }

      // v = momentum * v - lr * grad
      // param = param + v
      for (let i = 0; i < paramDense.data.length; i++) {
        velocity.data[i] = this.momentum * velocity.data[i] - this.learningRate * gradDense.data[i]
        paramDense.data[i] += velocity.data[i]
      }
    }
  }
}

export class AdamOptimizer {
  private learningRate: number
  private beta1: number
  private beta2: number
  private epsilon: number
  private t: number
  private m: Map<string, DenseTensor>
  private v: Map<string, DenseTensor>

  constructor(
    learningRate = 0.001,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
  ) {
    this.learningRate = learningRate
    this.beta1 = beta1
    this.beta2 = beta2
    this.epsilon = epsilon
    this.t = 0
    this.m = new Map()
    this.v = new Map()
  }

  step(parameters: Map<string, Tensor>, gradients: Map<string, Tensor>): void {
    this.t++

    for (const [name, param] of parameters) {
      const grad = gradients.get(name)
      if (!grad)
        continue

      const paramDense = param.type === 'sparse' ? toDense(param) : param
      const gradDense = grad.type === 'sparse' ? toDense(grad) : grad

      let m = this.m.get(name)
      let v = this.v.get(name)

      if (!m) {
        m = zeros(paramDense.shape)
        this.m.set(name, m)
      }
      if (!v) {
        v = zeros(paramDense.shape)
        this.v.set(name, v)
      }

      for (let i = 0; i < paramDense.data.length; i++) {
        const g = gradDense.data[i]

        // Update biased first moment estimate
        m.data[i] = this.beta1 * m.data[i] + (1 - this.beta1) * g
        // Update biased second raw moment estimate
        v.data[i] = this.beta2 * v.data[i] + (1 - this.beta2) * g * g

        // Compute bias-corrected estimates
        const mHat = m.data[i] / (1 - this.beta1 ** this.t)
        const vHat = v.data[i] / (1 - this.beta2 ** this.t)

        // Update parameters
        paramDense.data[i] -= this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon)
      }
    }
  }
}

/**
 * Create a differentiable engine
 */
export function createDifferentiableEngine(programOrSource: Program | string): DifferentiableEngine {
  return new DifferentiableEngine(programOrSource)
}

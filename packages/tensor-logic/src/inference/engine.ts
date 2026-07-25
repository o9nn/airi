/**
 * Tensor Logic Inference Engine
 *
 * Implements:
 * - Forward chaining: Execute equations sequentially until fixpoint
 * - Backward chaining: Treat equations as functions, recursively query
 *
 * This is the core execution engine for tensor logic programs
 */

import type { DenseTensor, Tensor, TensorShape } from '../core/types'
import type {
  BinaryOp,
  Expression,
  FunctionCall,
  NumberLiteral,
  Program,
  TensorEquation,
  TensorRef,
} from '../parser/ast'

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
  prelu,
  reciprocal,
  relu,
  selu,
  sigmoid,
  sigmoidTemperature,
  sin,
  smoothStep,
  softmax,
  sqrt,
  square,
  step,
  swish,
  tanh,
} from '../core/nonlinearities'
import {
  add,
  divide,
  join,
  project,
  subtract,
} from '../core/operations'
import {
  createDenseTensor,
  createShape,
  toDense,
} from '../core/types'
import { parse } from '../parser/parser'

/**
 * Tensor store: maps tensor names to tensors
 */
export type TensorStore = Map<string, Tensor>

/**
 * Read an axis name from a function call's argument list.
 *
 * An axis argument (`softmax(X, i)`) names an index variable, not a tensor, so
 * it has to be read off the syntax tree. Evaluating it would look up a tensor
 * called `i`, find nothing, and abort the whole call.
 */
function axisArg(call: FunctionCall, position: number): string | undefined {
  const arg = call.arguments[position]
  return arg?.type === 'tensor' ? arg.name : undefined
}

/**
 * Rebind a tensor's index names to the variables written at a reference site.
 *
 * The index variables in an equation — not the names a tensor happens to carry
 * in the store — decide which dimensions are contracted. Without this,
 * `Parent[x,y] * Parent[y,z]` would see two operands with identical index names
 * and sum over every dimension, collapsing the grandparent join to a scalar.
 *
 * This is a pure rename: element order is untouched, so the data is shared
 * rather than copied. The stored tensor is never mutated.
 */
function bindIndices(tensor: Tensor, ref: TensorRef): Tensor {
  // A bare reference (`X`) inherits the stored names; a rank mismatch means the
  // reference cannot be positionally aligned, so leave it alone.
  if (ref.indices.length === 0 || ref.indices.length !== tensor.shape.indices.length) {
    return tensor
  }

  const bound = createShape(tensor.shape.indices.map((idx, i) => ({
    ...idx,
    name: ref.indices[i].name,
  })))

  return { ...tensor, shape: bound }
}

/**
 * Shape registry: maps tensor names to shapes (for type checking)
 */
export type ShapeRegistry = Map<string, TensorShape>

/**
 * Execution options
 */
export interface ExecutionOptions {
  /** Maximum iterations for forward chaining fixpoint */
  maxIterations?: number
  /** Convergence threshold for fixpoint detection */
  convergenceThreshold?: number
  /** Whether to track execution history */
  trackHistory?: boolean
  /** Random seed for dropout etc. */
  randomSeed?: number
  /** Training mode (affects dropout, batchnorm) */
  training?: boolean
}

/**
 * Execution result
 */
export interface ExecutionResult {
  /** Final tensor store */
  tensors: TensorStore
  /** Number of iterations (for forward chaining) */
  iterations?: number
  /** Whether fixpoint was reached */
  converged?: boolean
  /** Execution history (if tracked) */
  history?: Array<{ step: number, tensor: string, values: number[] }>
}

/**
 * Forward chaining inference engine
 *
 * Executes tensor equations in order, repeating until fixpoint
 */
export class ForwardChainingEngine {
  private program: Program
  private store: TensorStore
  private shapes: ShapeRegistry
  private options: ExecutionOptions

  constructor(
    programOrSource: Program | string,
    initialTensors: TensorStore = new Map(),
    options: ExecutionOptions = {},
  ) {
    this.program = typeof programOrSource === 'string' ? parse(programOrSource) : programOrSource
    this.store = new Map(initialTensors)
    this.shapes = new Map()
    this.options = {
      maxIterations: 100,
      convergenceThreshold: 1e-6,
      trackHistory: false,
      training: true,
      ...options,
    }

    // Build shape registry from initial tensors
    for (const [name, tensor] of this.store) {
      this.shapes.set(name, tensor.shape)
    }
  }

  /**
   * Execute the program using forward chaining
   */
  execute(): ExecutionResult {
    const history: ExecutionResult['history'] = this.options.trackHistory ? [] : undefined
    let iterations = 0
    let converged = false

    while (iterations < this.options.maxIterations!) {
      iterations++
      let changed = false

      for (const stmt of this.program.statements) {
        if (stmt.type === 'equation') {
          const oldValue = this.store.get(stmt.lhs.name)
          const newValue = this.evaluateEquation(stmt)

          if (newValue) {
            // Check for convergence
            if (oldValue && oldValue.type === 'dense' && newValue.type === 'dense') {
              const diff = this.computeDifference(oldValue as DenseTensor, newValue)
              if (diff > this.options.convergenceThreshold!) {
                changed = true
              }
            }
            else {
              changed = true
            }

            this.store.set(stmt.lhs.name, newValue)

            if (history) {
              history.push({
                step: iterations,
                tensor: stmt.lhs.name,
                values: [...(newValue.type === 'dense' ? newValue.data : toDense(newValue).data)],
              })
            }
          }
        }
        else if (stmt.type === 'rule') {
          // Convert Datalog rule to tensor equation and evaluate
          const tensorEq = this.datalogToTensor(stmt)
          if (tensorEq) {
            const newValue = this.evaluateEquation(tensorEq)
            if (newValue) {
              this.store.set(stmt.head.name, newValue)
            }
          }
        }
      }

      if (!changed) {
        converged = true
        break
      }
    }

    return {
      tensors: this.store,
      iterations,
      converged,
      history,
    }
  }

  /**
   * Execute a single step (one pass through all equations)
   */
  step(): void {
    for (const stmt of this.program.statements) {
      if (stmt.type === 'equation') {
        const newValue = this.evaluateEquation(stmt)
        if (newValue) {
          this.store.set(stmt.lhs.name, newValue)
        }
      }
    }
  }

  /**
   * Get a tensor from the store
   */
  getTensor(name: string): Tensor | undefined {
    return this.store.get(name)
  }

  /**
   * Set a tensor in the store
   */
  setTensor(name: string, tensor: Tensor): void {
    this.store.set(name, tensor)
    this.shapes.set(name, tensor.shape)
  }

  /**
   * Evaluate a tensor equation
   */
  private evaluateEquation(eq: TensorEquation): Tensor | null {
    try {
      const rhsValue = this.evaluateExpression(eq.rhs)

      if (!rhsValue)
        return null

      // Apply normalization if specified
      if (eq.normalizeIndices && eq.normalizeIndices.length > 0) {
        return softmax(rhsValue, eq.normalizeIndices[0])
      }

      // Project to LHS indices if different
      const lhsIndices = eq.lhs.indices.map(i => i.name)
      if (lhsIndices.length > 0 && rhsValue.indexNames.length > 0) {
        const rhsIndices = new Set(rhsValue.indexNames)
        const needsProjection = lhsIndices.some(i => !rhsIndices.has(i))
          || rhsValue.indexNames.some(i => !lhsIndices.includes(i))

        if (needsProjection && lhsIndices.length < rhsValue.indexNames.length) {
          return project(rhsValue, lhsIndices)
        }
      }

      return rhsValue
    }
    catch {
      return null
    }
  }

  /**
   * Evaluate an expression
   */
  private evaluateExpression(expr: Expression): Tensor | null {
    switch (expr.type) {
      case 'number':
        return this.evaluateNumber(expr)

      case 'tensor':
        return this.evaluateTensorRef(expr)

      case 'binary':
        return this.evaluateBinaryOp(expr)

      case 'unary':
        return expr.operator === '-'
          ? neg(this.evaluateExpression(expr.operand)!)
          : null

      case 'function':
        return this.evaluateFunctionCall(expr)

      case 'relation':
        // Relations are treated as tensor lookups
        return this.store.get(expr.name) ?? null

      default:
        return null
    }
  }

  /**
   * Evaluate a number literal
   */
  private evaluateNumber(num: NumberLiteral): Tensor {
    return createDenseTensor(createShape([]), [num.value])
  }

  /**
   * Evaluate a tensor reference
   */
  private evaluateTensorRef(ref: TensorRef): Tensor | null {
    const tensor = this.store.get(ref.name)
    if (!tensor)
      return null

    return bindIndices(tensor, ref)
  }

  /**
   * Evaluate a binary operation
   */
  private evaluateBinaryOp(op: BinaryOp): Tensor | null {
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
        // Implicit join (Einstein summation)
        return join(left, right)
      case '/':
        return divide(left, right)
      case '^':
        // Power - right side should be a scalar
        if (right.type === 'dense' && right.shape.size === 1) {
          return pow(left, right.data[0])
        }
        return null
      default:
        return null
    }
  }

  /**
   * Evaluate a function call
   */
  private evaluateFunctionCall(call: FunctionCall): Tensor | null {
    const args = call.arguments.map(arg => this.evaluateExpression(arg))

    // Only the operand has to resolve to a tensor. Trailing arguments are
    // either scalars (which evaluate fine) or axis names (which deliberately
    // do not), and each call site below already falls back to a default.
    if (args[0] == null)
      return null

    const firstArg = args[0]

    // Map function names to implementations
    switch (call.name.toLowerCase()) {
      case 'step':
      case 'h':
      case 'heaviside':
        return step(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 0)

      case 'smoothstep':
        return smoothStep(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 10)

      case 'sigmoid':
      case 'sig':
      case 'σ':
        return sigmoid(firstArg)

      case 'sigmoid_t':
        return sigmoidTemperature(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 1)

      case 'relu':
        return relu(firstArg)

      case 'leaky_relu':
        return leakyRelu(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 0.01)

      case 'elu':
        return elu(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 1)

      case 'selu':
        return selu(firstArg)

      case 'gelu':
        return gelu(firstArg)

      case 'swish':
      case 'silu':
        return swish(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 1)

      case 'tanh':
        return tanh(firstArg)

      case 'softmax':
        return softmax(firstArg, axisArg(call, 1))

      case 'log_softmax':
        return logSoftmax(firstArg)

      case 'lnorm':
      case 'layernorm':
        return lnorm(firstArg)

      case 'batchnorm':
        return batchNorm(firstArg, axisArg(call, 1) ?? 'b')

      case 'dropout':
        return dropout(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 0.5, this.options.training)

      case 'abs':
        return abs(firstArg)

      case 'square':
      case 'sq':
        return square(firstArg)

      case 'sqrt':
        return sqrt(firstArg)

      case 'log':
      case 'ln':
        return log(firstArg)

      case 'exp':
        return exp(firstArg)

      case 'sin':
        return sin(firstArg)

      case 'cos':
        return cos(firstArg)

      case 'neg':
        return neg(firstArg)

      case 'reciprocal':
      case 'inv':
        return reciprocal(firstArg)

      case 'clamp':
        return clamp(
          firstArg,
          args[1] ? (args[1] as DenseTensor).data[0] : 0,
          args[2] ? (args[2] as DenseTensor).data[0] : 1,
        )

      case 'prelu':
        return prelu(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 0.01)

      default:
        // Unknown function - return null
        console.warn(`Unknown function: ${call.name}`)
        return null
    }
  }

  /**
   * Convert a Datalog rule to a tensor equation
   */
  private datalogToTensor(rule: import('../parser/ast').DatalogRule): TensorEquation | null {
    // A[x,z] = H(S[x,y] * P[y,z])
    // Convert: Head(x,z) <- Body1(x,y), Body2(y,z)

    const headVars = rule.head.arguments
    const bodyExprs: Expression[] = rule.body.map(rel => ({
      type: 'tensor' as const,
      name: rel.name,
      indices: rel.arguments.map(v => ({ type: 'index' as const, name: v })),
    }))

    // Combine body expressions with multiplication (join)
    let combinedBody: Expression = bodyExprs[0]
    for (let i = 1; i < bodyExprs.length; i++) {
      combinedBody = {
        type: 'binary',
        operator: '*',
        left: combinedBody,
        right: bodyExprs[i],
      }
    }

    // Wrap with step function
    const rhsExpr: Expression = {
      type: 'function',
      name: 'step',
      arguments: [combinedBody],
    }

    return {
      type: 'equation',
      lhs: {
        type: 'tensor',
        name: rule.head.name,
        indices: headVars.map(v => ({ type: 'index', name: v })),
      },
      rhs: rhsExpr,
    }
  }

  /**
   * Compute L2 difference between two tensors
   */
  private computeDifference(a: DenseTensor, b: DenseTensor): number {
    if (a.data.length !== b.data.length)
      return Number.POSITIVE_INFINITY

    let sum = 0
    for (let i = 0; i < a.data.length; i++) {
      const diff = a.data[i] - b.data[i]
      sum += diff * diff
    }
    return Math.sqrt(sum / a.data.length)
  }
}

/**
 * Backward chaining inference engine
 *
 * Treats equations as functions, recursively queries required tensors
 */
export class BackwardChainingEngine {
  private program: Program
  private store: TensorStore
  private shapes: ShapeRegistry
  private equationMap: Map<string, TensorEquation>
  private computing: Set<string>

  constructor(
    programOrSource: Program | string,
    initialTensors: TensorStore = new Map(),
  ) {
    this.program = typeof programOrSource === 'string' ? parse(programOrSource) : programOrSource
    this.store = new Map(initialTensors)
    this.shapes = new Map()
    this.computing = new Set()

    // Build equation index
    this.equationMap = new Map()
    for (const stmt of this.program.statements) {
      if (stmt.type === 'equation') {
        this.equationMap.set(stmt.lhs.name, stmt)
      }
    }

    // Build shape registry from initial tensors
    for (const [name, tensor] of this.store) {
      this.shapes.set(name, tensor.shape)
    }
  }

  /**
   * Query a tensor by name (backward chaining)
   */
  query(tensorName: string): Tensor | null {
    // Check if already computed
    if (this.store.has(tensorName)) {
      return this.store.get(tensorName)!
    }

    // Check for circular dependency
    if (this.computing.has(tensorName)) {
      // Return zero tensor as default (unresolvable subquery)
      return null
    }

    // Find equation for this tensor
    const eq = this.equationMap.get(tensorName)
    if (!eq)
      return null

    // Mark as computing to detect cycles
    this.computing.add(tensorName)

    try {
      // Recursively evaluate RHS
      const result = this.evaluateExpression(eq.rhs)

      if (result) {
        this.store.set(tensorName, result)
      }

      return result
    }
    finally {
      this.computing.delete(tensorName)
    }
  }

  /**
   * Get a tensor from the store (without computing)
   */
  getTensor(name: string): Tensor | undefined {
    return this.store.get(name)
  }

  /**
   * Set a tensor in the store
   */
  setTensor(name: string, tensor: Tensor): void {
    this.store.set(name, tensor)
    this.shapes.set(name, tensor.shape)
  }

  /**
   * Evaluate an expression, recursively querying dependencies
   */
  private evaluateExpression(expr: Expression): Tensor | null {
    switch (expr.type) {
      case 'number':
        return createDenseTensor(createShape([]), [expr.value])

      case 'tensor': {
        // Recursively query the tensor, then bind it to the index variables
        // used at this reference site.
        const result = this.query(expr.name)
        return result === null ? null : bindIndices(result, expr)
      }

      case 'binary': {
        const left = this.evaluateExpression(expr.left)
        const right = this.evaluateExpression(expr.right)

        if (!left || !right)
          return null

        switch (expr.operator) {
          case '+':
            return add(left, right)
          case '-':
            return subtract(left, right)
          case '*':
            return join(left, right)
          case '/':
            return divide(left, right)
          case '^':
            if (right.type === 'dense' && right.shape.size === 1) {
              return pow(left, right.data[0])
            }
            return null
          default:
            return null
        }
      }

      case 'unary':
        return expr.operator === '-'
          ? neg(this.evaluateExpression(expr.operand)!)
          : null

      case 'function':
        return this.evaluateFunctionCall(expr)

      case 'relation':
        return this.query(expr.name)

      default:
        return null
    }
  }

  /**
   * Evaluate a function call
   */
  private evaluateFunctionCall(call: FunctionCall): Tensor | null {
    const args = call.arguments.map(arg => this.evaluateExpression(arg))

    // Only the operand has to resolve to a tensor. Trailing arguments are
    // either scalars (which evaluate fine) or axis names (which deliberately
    // do not), and each call site below already falls back to a default.
    if (args[0] == null)
      return null

    const firstArg = args[0]

    // Same function mapping as forward chaining
    switch (call.name.toLowerCase()) {
      case 'step':
      case 'h':
        return step(firstArg, args[1] ? (args[1] as DenseTensor).data[0] : 0)
      case 'sigmoid':
      case 'sig':
        return sigmoid(firstArg)
      case 'relu':
        return relu(firstArg)
      case 'softmax':
        return softmax(firstArg)
      case 'lnorm':
        return lnorm(firstArg)
      case 'tanh':
        return tanh(firstArg)
      case 'exp':
        return exp(firstArg)
      case 'log':
        return log(firstArg)
      default:
        return null
    }
  }
}

/**
 * Create a forward chaining engine
 */
export function createForwardEngine(
  programOrSource: Program | string,
  initialTensors?: TensorStore,
  options?: ExecutionOptions,
): ForwardChainingEngine {
  return new ForwardChainingEngine(programOrSource, initialTensors, options)
}

/**
 * Create a backward chaining engine
 */
export function createBackwardEngine(
  programOrSource: Program | string,
  initialTensors?: TensorStore,
): BackwardChainingEngine {
  return new BackwardChainingEngine(programOrSource, initialTensors)
}

/**
 * Execute a tensor logic program with forward chaining
 */
export function execute(
  programOrSource: Program | string,
  initialTensors?: TensorStore,
  options?: ExecutionOptions,
): ExecutionResult {
  const engine = createForwardEngine(programOrSource, initialTensors, options)
  return engine.execute()
}

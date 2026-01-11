/**
 * @proj-airi/tensor-logic
 *
 * Tensor Logic: An AI programming language unifying deep learning and symbolic AI
 *
 * Based on the insight that logical rules and Einstein summation are essentially
 * the same operation, tensor logic provides a unified framework for:
 * - Neural networks (transformers, MLPs, RNNs, GNNs)
 * - Symbolic reasoning (Datalog, forward/backward chaining)
 * - Probabilistic graphical models
 * - Kernel machines
 * - Embedded reasoning with analogical inference
 *
 * Key features:
 * - Tensor equations as the sole language construct
 * - Einstein summation for join/projection operations
 * - Automatic differentiation for learning
 * - Tucker decomposition for efficient embedding-based reasoning
 * - Temperature-controlled analogical reasoning
 *
 * @see https://tensor-logic.org/
 * @see https://arxiv.org/abs/2510.12269
 */

// Core tensor types and operations
export * from './core'

// Parser for tensor logic equations
export * from './parser'

// Inference engines (forward and backward chaining)
export * from './inference'

// Automatic differentiation
export * from './autodiff'

// Embeddings and analogical reasoning
export * from './embeddings'

// Re-export commonly used types and functions at top level
export {
  // Types
  createDenseTensor,
  createRelation,
  createShape,
  createSparseTensor,
  // Operations
  einsum,
  join,
  project,
  // Nonlinearities
  relu,
  sigmoid,
  softmax,
  step,
} from './core'

export {
  // Parser
  parse,
  parseEquation,
} from './parser'

export {
  // Inference
  execute,
  ForwardChainingEngine,
} from './inference'

export {
  // Autodiff
  DifferentiableEngine,
  LossFunctions,
} from './autodiff'

export {
  // Embeddings
  AnalogicalReasoningEngine,
  tuckerDecompose,
} from './embeddings'

/**
 * Version of the tensor-logic package
 */
export const VERSION = '0.1.0'

/**
 * Quick start example:
 *
 * ```typescript
 * import {
 *   createDenseTensor,
 *   createShape,
 *   parse,
 *   execute
 * } from '@proj-airi/tensor-logic';
 *
 * // Define tensors
 * const W = createDenseTensor(
 *   createShape([{ name: 'i', size: 3 }]),
 *   [1, 2, 3]
 * );
 * const X = createDenseTensor(
 *   createShape([{ name: 'i', size: 3 }]),
 *   [0.1, 0.2, 0.3]
 * );
 *
 * // Define program
 * const program = parse('Y = sigmoid(W[i] * X[i])');
 *
 * // Execute
 * const result = execute(program, new Map([['W', W], ['X', X]]));
 * console.log(result.tensors.get('Y'));
 * ```
 */

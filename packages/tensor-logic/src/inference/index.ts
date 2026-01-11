/**
 * Tensor Logic Inference Module
 *
 * Exports forward and backward chaining inference engines
 */

export type {
  ExecutionOptions,
  ExecutionResult,
  ShapeRegistry,
  TensorStore,
} from './engine'

export {
  BackwardChainingEngine,
  createBackwardEngine,
  createForwardEngine,
  execute,
  ForwardChainingEngine,
} from './engine'

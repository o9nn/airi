/**
 * Tensor Logic Automatic Differentiation Module
 *
 * Exports differentiable engine, loss functions, and optimizers
 */

export type { TapeEntry } from './autodiff'

export {
  AdamOptimizer,
  createDifferentiableEngine,
  DifferentiableEngine,
  GradientTape,
  LossFunctions,
  SGDOptimizer,
} from './autodiff'

/**
 * Tensor Logic Embeddings Module
 *
 * Exports Tucker decomposition and analogical reasoning components
 */

// Analogical Reasoning
export type { ReasoningMode } from './analogical'

export {
  AnalogicalReasoningEngine,
  analogyCompletion,
  computeOptimalTemperature,
  computeSimilarityMatrix,
  createAnalogicalEngine,
  multiHopAnalogy,
  ReasoningModes,
  temperatureSigmoid,
} from './analogical'

// Tucker Decomposition
export type { TuckerDecomposition } from './tucker'

export {
  createRandomEmbeddings,
  embedRelation,
  queryEmbeddedRelation,
  querySuperposition,
  superpositionEncode,
  tuckerDecompose,
  tuckerError,
  tuckerReconstruct,
} from './tucker'

/**
 * Tensor Logic Embeddings Module
 *
 * Exports Tucker decomposition and analogical reasoning components
 */

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

// Analogical Reasoning
export type { ReasoningMode } from './analogical'

export {
  analogyCompletion,
  AnalogicalReasoningEngine,
  computeOptimalTemperature,
  computeSimilarityMatrix,
  createAnalogicalEngine,
  multiHopAnalogy,
  ReasoningModes,
  temperatureSigmoid,
} from './analogical'

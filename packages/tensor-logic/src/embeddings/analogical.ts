/**
 * Temperature-Controlled Analogical Reasoning
 *
 * Implements the paper's approach to combining deductive and analogical reasoning:
 * - Temperature T=0: Pure deductive reasoning (Gram matrix → identity)
 * - Temperature T>0: Increasing analogical reasoning (similar objects borrow inferences)
 *
 * Key insight: σ(Sim, T) where Sim is the Gram matrix of embeddings
 * allows smooth interpolation between symbolic and neural reasoning.
 */

import type { DenseTensor, SparseTensor, Tensor } from '../core/types'
import { join, project } from '../core/operations'
import {
  clone,
  coordsToFlat,
  createDenseTensor,
  createShape,
  flatToCoords,
  toDense,
} from '../core/types'

/**
 * Compute similarity matrix (Gram matrix) from embeddings
 *
 * Sim[x,x'] = Σ_d Emb[x,d] * Emb[x',d]
 */
export function computeSimilarityMatrix(embeddings: DenseTensor): DenseTensor {
  const numObjects = embeddings.shape.indices[0].size
  const embDim = embeddings.shape.indices[1].size

  const outShape = createShape([
    { name: 'x', size: numObjects },
    { name: 'x_prime', size: numObjects },
  ])
  const outData = new Array(numObjects * numObjects).fill(0)

  for (let x = 0; x < numObjects; x++) {
    for (let xp = 0; xp < numObjects; xp++) {
      let dot = 0
      for (let d = 0; d < embDim; d++) {
        dot += embeddings.data[x * embDim + d] * embeddings.data[xp * embDim + d]
      }
      outData[x * numObjects + xp] = dot
    }
  }

  return createDenseTensor(outShape, outData)
}

/**
 * Apply temperature-controlled sigmoid to similarity matrix
 *
 * σ(x, T) = 1 / (1 + e^(-x/T))
 *
 * At T=0: approaches step function (pure deductive)
 * At T→∞: approaches 0.5 everywhere (maximum uncertainty)
 */
export function temperatureSigmoid(
  similarityMatrix: DenseTensor,
  temperature: number,
): DenseTensor {
  if (temperature <= 0) {
    // Pure deductive: return identity-like matrix
    // High similarity (>0) → 1, low similarity (≤0) → 0
    return createDenseTensor(
      similarityMatrix.shape,
      similarityMatrix.data.map(x => (x > 0.5 ? 1 : 0)),
    )
  }

  return createDenseTensor(
    similarityMatrix.shape,
    similarityMatrix.data.map(x => 1 / (1 + Math.exp(-x / temperature))),
  )
}

/**
 * Analogical reasoning engine
 *
 * Uses temperature-controlled similarity to propagate inferences
 * from similar objects.
 */
export class AnalogicalReasoningEngine {
  private embeddings: DenseTensor
  private similarityMatrix: DenseTensor
  private temperature: number
  private transformedSimilarity: DenseTensor

  constructor(embeddings: DenseTensor, temperature = 1.0) {
    this.embeddings = embeddings
    this.temperature = temperature
    this.similarityMatrix = computeSimilarityMatrix(embeddings)
    this.transformedSimilarity = temperatureSigmoid(this.similarityMatrix, temperature)
  }

  /**
   * Set the temperature for analogical reasoning
   */
  setTemperature(temperature: number): void {
    this.temperature = temperature
    this.transformedSimilarity = temperatureSigmoid(this.similarityMatrix, temperature)
  }

  /**
   * Get current temperature
   */
  getTemperature(): number {
    return this.temperature
  }

  /**
   * Propagate beliefs using analogical reasoning
   *
   * Given beliefs B[x], compute new beliefs that incorporate
   * inferences from similar objects:
   *
   * B'[x] = Σ_x' Sim_T[x,x'] * B[x']
   */
  propagateBeliefs(beliefs: DenseTensor): DenseTensor {
    const numObjects = this.embeddings.shape.indices[0].size

    // B'[x] = Σ_x' Sim[x,x'] * B[x']
    const outShape = createShape([{ name: 'x', size: numObjects }])
    const outData = new Array(numObjects).fill(0)

    for (let x = 0; x < numObjects; x++) {
      for (let xp = 0; xp < numObjects; xp++) {
        outData[x] += this.transformedSimilarity.data[x * numObjects + xp] * beliefs.data[xp]
      }
    }

    return createDenseTensor(outShape, outData)
  }

  /**
   * Apply analogical reasoning to a relation
   *
   * Given R[x,y], compute R'[x,y] that incorporates similar objects:
   * R'[x,y] = Σ_{x',y'} Sim_T[x,x'] * R[x',y'] * Sim_T[y,y']
   */
  propagateRelation(relation: Tensor): DenseTensor {
    const dense = relation.type === 'sparse' ? toDense(relation) : relation
    const numObjects = this.embeddings.shape.indices[0].size

    const outShape = createShape([
      { name: 'x', size: numObjects },
      { name: 'y', size: numObjects },
    ])
    const outData = new Array(numObjects * numObjects).fill(0)

    // R'[x,y] = Σ_{x',y'} Sim[x,x'] * R[x',y'] * Sim[y,y']
    for (let x = 0; x < numObjects; x++) {
      for (let y = 0; y < numObjects; y++) {
        let sum = 0
        for (let xp = 0; xp < numObjects; xp++) {
          for (let yp = 0; yp < numObjects; yp++) {
            const simX = this.transformedSimilarity.data[x * numObjects + xp]
            const simY = this.transformedSimilarity.data[y * numObjects + yp]
            const relVal = dense.data[xp * numObjects + yp]
            sum += simX * relVal * simY
          }
        }
        outData[x * numObjects + y] = sum
      }
    }

    return createDenseTensor(outShape, outData)
  }

  /**
   * Query a relation with analogical reasoning
   *
   * Returns the degree to which R(x,y) holds, considering similar objects
   */
  queryRelation(relation: Tensor, x: number, y: number): number {
    const propagated = this.propagateRelation(relation)
    const numObjects = this.embeddings.shape.indices[0].size
    return propagated.data[x * numObjects + y]
  }

  /**
   * Infer new facts using analogical reasoning on rules
   *
   * Given a rule like "if R(x,y) then S(x,y)" and base facts,
   * use analogical reasoning to infer new facts.
   */
  inferWithAnalogy(
    baseFacts: Tensor,
    ruleApplication: (facts: DenseTensor) => DenseTensor,
  ): DenseTensor {
    // First, propagate base facts using similarity
    const propagatedFacts = this.propagateRelation(baseFacts)

    // Apply the rule to propagated facts
    const inferred = ruleApplication(propagatedFacts)

    // Optionally propagate again for second-order analogies
    // return this.propagateRelation(inferred);

    return inferred
  }

  /**
   * Compute analogical distance between two objects
   *
   * Lower distance = more similar = more likely to share properties
   */
  analogicalDistance(x: number, y: number): number {
    const numObjects = this.embeddings.shape.indices[0].size
    const similarity = this.transformedSimilarity.data[x * numObjects + y]
    // Convert similarity to distance
    return 1 - similarity
  }

  /**
   * Find the k most similar objects to a given object
   */
  findMostSimilar(x: number, k: number): Array<{ object: number, similarity: number }> {
    const numObjects = this.embeddings.shape.indices[0].size
    const similarities: Array<{ object: number, similarity: number }> = []

    for (let xp = 0; xp < numObjects; xp++) {
      if (xp !== x) {
        similarities.push({
          object: xp,
          similarity: this.transformedSimilarity.data[x * numObjects + xp],
        })
      }
    }

    // Sort by similarity descending
    similarities.sort((a, b) => b.similarity - a.similarity)

    return similarities.slice(0, k)
  }

  /**
   * Get similarity between two objects
   */
  getSimilarity(x: number, y: number): number {
    const numObjects = this.embeddings.shape.indices[0].size
    return this.transformedSimilarity.data[x * numObjects + y]
  }
}

/**
 * Reasoning spectrum: interpolate between pure deduction and pure analogy
 */
export interface ReasoningMode {
  /** Mode name */
  name: string
  /** Temperature value */
  temperature: number
  /** Description */
  description: string
}

export const ReasoningModes: Record<string, ReasoningMode> = {
  /** Pure symbolic deduction: only exact matches count */
  DEDUCTIVE: {
    name: 'Deductive',
    temperature: 0,
    description: 'Pure symbolic reasoning - only exact matches',
  },

  /** Weak analogy: slight borrowing from similar objects */
  WEAK_ANALOGY: {
    name: 'Weak Analogy',
    temperature: 0.1,
    description: 'Mostly deductive with slight analogical influence',
  },

  /** Moderate analogy: balanced deduction and analogy */
  MODERATE_ANALOGY: {
    name: 'Moderate Analogy',
    temperature: 0.5,
    description: 'Balanced deductive and analogical reasoning',
  },

  /** Strong analogy: heavy analogical reasoning */
  STRONG_ANALOGY: {
    name: 'Strong Analogy',
    temperature: 1.0,
    description: 'Strong analogical influence',
  },

  /** Maximum uncertainty: all objects equally similar */
  MAXIMUM_UNCERTAINTY: {
    name: 'Maximum Uncertainty',
    temperature: 10.0,
    description: 'Maximum uncertainty - all objects treated similarly',
  },
}

/**
 * Create an analogical reasoning engine with a predefined mode
 */
export function createAnalogicalEngine(
  embeddings: DenseTensor,
  mode: ReasoningMode = ReasoningModes.MODERATE_ANALOGY,
): AnalogicalReasoningEngine {
  return new AnalogicalReasoningEngine(embeddings, mode.temperature)
}

/**
 * Compute the optimal temperature for a given task
 *
 * This heuristic balances the tradeoff between:
 * - Low temperature: more precise but limited generalization
 * - High temperature: more generalization but noisier
 */
export function computeOptimalTemperature(
  embeddings: DenseTensor,
  knownFacts: Tensor,
  validationFacts: Tensor,
  temperatureRange: number[] = [0.01, 0.1, 0.5, 1.0, 2.0],
): number {
  const dense = knownFacts.type === 'sparse' ? toDense(knownFacts) : knownFacts
  const validation = validationFacts.type === 'sparse' ? toDense(validationFacts) : validationFacts

  let bestTemperature = temperatureRange[0]
  let bestScore = Number.NEGATIVE_INFINITY

  for (const temp of temperatureRange) {
    const engine = new AnalogicalReasoningEngine(embeddings, temp)
    const propagated = engine.propagateRelation(dense)

    // Compute score against validation facts
    let score = 0
    for (let i = 0; i < validation.data.length; i++) {
      if (validation.data[i] > 0.5) {
        score += propagated.data[i] // True positive reward
      }
      else {
        score -= propagated.data[i] * 0.5 // False positive penalty
      }
    }

    if (score > bestScore) {
      bestScore = score
      bestTemperature = temp
    }
  }

  return bestTemperature
}

/**
 * Multi-hop analogical reasoning
 *
 * Applies analogical propagation multiple times to discover
 * transitive analogies.
 */
export function multiHopAnalogy(
  engine: AnalogicalReasoningEngine,
  initialFacts: Tensor,
  hops: number,
): DenseTensor {
  let current = initialFacts.type === 'sparse' ? toDense(initialFacts) : clone(initialFacts)

  for (let i = 0; i < hops; i++) {
    current = engine.propagateRelation(current)
  }

  return current
}

/**
 * Compute analogy completion scores
 *
 * Given A:B::C:? find the best D
 * Uses the parallelogram rule in embedding space
 */
export function analogyCompletion(
  embeddings: DenseTensor,
  objectA: number,
  objectB: number,
  objectC: number,
): Array<{ object: number, score: number }> {
  const embDim = embeddings.shape.indices[1].size
  const numObjects = embeddings.shape.indices[0].size

  // Compute A - B + C in embedding space
  const targetEmb: number[] = []
  for (let d = 0; d < embDim; d++) {
    const a = embeddings.data[objectA * embDim + d]
    const b = embeddings.data[objectB * embDim + d]
    const c = embeddings.data[objectC * embDim + d]
    targetEmb.push(c + (b - a))
  }

  // Find closest objects to target
  const scores: Array<{ object: number, score: number }> = []

  for (let obj = 0; obj < numObjects; obj++) {
    if (obj === objectA || obj === objectB || obj === objectC)
      continue

    let dot = 0
    let normTarget = 0
    let normObj = 0

    for (let d = 0; d < embDim; d++) {
      const t = targetEmb[d]
      const o = embeddings.data[obj * embDim + d]
      dot += t * o
      normTarget += t * t
      normObj += o * o
    }

    const similarity = dot / (Math.sqrt(normTarget) * Math.sqrt(normObj) + 1e-10)
    scores.push({ object: obj, score: similarity })
  }

  scores.sort((a, b) => b.score - a.score)
  return scores
}

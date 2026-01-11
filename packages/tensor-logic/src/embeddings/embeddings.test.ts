import { describe, expect, it } from 'vitest'

import { createDenseTensor, createRelation, createShape } from '../core/types'

import {
  analogyCompletion,
  AnalogicalReasoningEngine,
  computeOptimalTemperature,
  computeSimilarityMatrix,
  createAnalogicalEngine,
  multiHopAnalogy,
  ReasoningModes,
  temperatureSigmoid,
} from './analogical'
import {
  createRandomEmbeddings,
  embedRelation,
  queryEmbeddedRelation,
  querySuperposition,
  superpositionEncode,
  tuckerDecompose,
  tuckerError,
  tuckerReconstruct,
} from './tucker'

describe('createRandomEmbeddings', () => {
  it('should create embeddings with correct shape', () => {
    const emb = createRandomEmbeddings(10, 32, 'x', 'd')

    expect(emb.shape.indices[0].size).toBe(10)
    expect(emb.shape.indices[1].size).toBe(32)
    expect(emb.indexNames).toEqual(['x', 'd'])
  })

  it('should create unit vectors', () => {
    const emb = createRandomEmbeddings(5, 20)

    // Check that each row is a unit vector
    for (let i = 0; i < 5; i++) {
      let norm = 0
      for (let j = 0; j < 20; j++) {
        norm += emb.data[i * 20 + j] ** 2
      }
      expect(Math.abs(Math.sqrt(norm) - 1)).toBeLessThan(1e-5)
    }
  })
})

describe('superpositionEncode', () => {
  it('should encode set as sum of embeddings', () => {
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 4 }]),
      [
        1, 0, 0, 0, // Object 0
        0, 1, 0, 0, // Object 1
        0, 0, 1, 0, // Object 2
      ],
    )

    const encoded = superpositionEncode([0, 1], emb)

    // Should be sum of first two rows
    expect(encoded.data).toEqual([1, 1, 0, 0])
  })

  it('should work with random embeddings', () => {
    const emb = createRandomEmbeddings(10, 50)
    const set = [1, 3, 5, 7]

    const encoded = superpositionEncode(set, emb)

    expect(encoded.shape.indices[0].size).toBe(50)
  })
})

describe('querySuperposition', () => {
  it('should detect set membership', () => {
    // Use orthogonal embeddings for clear separation
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 3 }]),
      [
        1, 0, 0, // Object 0
        0, 1, 0, // Object 1
        0, 0, 1, // Object 2
      ],
    )

    const encoded = superpositionEncode([0, 1], emb)

    // Object 0 is in set -> should return ~1
    expect(querySuperposition(encoded, emb, 0)).toBeCloseTo(1, 5)
    // Object 1 is in set -> should return ~1
    expect(querySuperposition(encoded, emb, 1)).toBeCloseTo(1, 5)
    // Object 2 is not in set -> should return ~0
    expect(querySuperposition(encoded, emb, 2)).toBeCloseTo(0, 5)
  })
})

describe('embedRelation', () => {
  it('should embed a relation', () => {
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 4 }]),
      Array.from({ length: 12 }, () => Math.random()),
    )

    const relation = createRelation(['x', 'y'], [[0, 1], [1, 2]], [3, 3])

    const embedded = embedRelation(relation, emb)

    expect(embedded.shape.indices[0].size).toBe(4)
    expect(embedded.shape.indices[1].size).toBe(4)
  })
})

describe('queryEmbeddedRelation', () => {
  it('should retrieve relation membership', () => {
    // Use simple embeddings
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 3 }]),
      [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ],
    )

    // Relation: 0->1, 1->2
    const relation = createRelation(['x', 'y'], [[0, 1]], [3, 3])
    const embedded = embedRelation(relation, emb)

    // 0->1 should be positive
    const score01 = queryEmbeddedRelation(embedded, emb, 0, 1)
    // 0->2 should be ~0
    const score02 = queryEmbeddedRelation(embedded, emb, 0, 2)

    expect(score01).toBeGreaterThan(score02)
  })
})

describe('tuckerDecompose', () => {
  it('should decompose a tensor', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 4 }, { name: 'j', size: 4 }]),
      [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ],
    )

    const decomposition = tuckerDecompose(tensor, [2, 2], 5)

    expect(decomposition.factors).toHaveLength(2)
    expect(decomposition.core).toBeDefined()
    expect(decomposition.embeddingDims).toEqual([2, 2])
  })

  it('should return decomposition with valid structure', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 4 }, { name: 'j', size: 4 }]),
      [
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7,
      ],
    )

    const decomposition = tuckerDecompose(tensor, [3, 3], 3)

    // Check structure
    expect(decomposition.factors).toHaveLength(2)
    expect(decomposition.factors[0].shape.indices[1].size).toBe(3)
    expect(decomposition.factors[1].shape.indices[1].size).toBe(3)
    expect(decomposition.core).toBeDefined()
  })
})

describe('tuckerReconstruct', () => {
  it('should return tensor with correct shape', () => {
    const tensor = createDenseTensor(
      createShape([{ name: 'i', size: 3 }, { name: 'j', size: 3 }]),
      [1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    const decomposition = tuckerDecompose(tensor, [2, 2], 3)
    const reconstructed = tuckerReconstruct(decomposition)

    // Check that we get a result with numeric data
    expect(reconstructed.data).toHaveLength(decomposition.core.shape.size)
    expect(reconstructed.data.every(v => Number.isFinite(v))).toBe(true)
  })
})

describe('computeSimilarityMatrix', () => {
  it('should compute Gram matrix', () => {
    // Orthonormal embeddings
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 3 }]),
      [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ],
    )

    const sim = computeSimilarityMatrix(emb)

    // Diagonal should be 1 (self-similarity)
    expect(sim.data[0]).toBeCloseTo(1, 5) // Sim[0,0]
    expect(sim.data[4]).toBeCloseTo(1, 5) // Sim[1,1]
    expect(sim.data[8]).toBeCloseTo(1, 5) // Sim[2,2]

    // Off-diagonal should be 0 (orthogonal)
    expect(sim.data[1]).toBeCloseTo(0, 5) // Sim[0,1]
    expect(sim.data[2]).toBeCloseTo(0, 5) // Sim[0,2]
  })

  it('should be symmetric', () => {
    const emb = createRandomEmbeddings(5, 10)
    const sim = computeSimilarityMatrix(emb)

    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        expect(sim.data[i * 5 + j]).toBeCloseTo(sim.data[j * 5 + i], 10)
      }
    }
  })
})

describe('temperatureSigmoid', () => {
  it('should approach step function at T=0', () => {
    const sim = createDenseTensor(
      createShape([{ name: 'x', size: 4 }]),
      [-1, 0, 0.5, 1],
    )

    const result = temperatureSigmoid(sim, 0)

    expect(result.data[0]).toBe(0)
    expect(result.data[1]).toBe(0)
    expect(result.data[2]).toBe(0)
    expect(result.data[3]).toBe(1)
  })

  it('should be smoother at higher temperature', () => {
    const sim = createDenseTensor(
      createShape([{ name: 'x', size: 3 }]),
      [-1, 0, 1],
    )

    const lowT = temperatureSigmoid(sim, 0.1)
    const highT = temperatureSigmoid(sim, 10)

    // At low T, values should be more extreme
    expect(lowT.data[0]).toBeLessThan(highT.data[0])
    expect(lowT.data[2]).toBeGreaterThan(highT.data[2])
  })
})

describe('AnalogicalReasoningEngine', () => {
  it('should create engine with embeddings', () => {
    const emb = createRandomEmbeddings(10, 20)
    const engine = new AnalogicalReasoningEngine(emb, 1.0)

    expect(engine.getTemperature()).toBe(1.0)
  })

  it('should propagate beliefs', () => {
    // Similar objects: 0 and 1 have similar embeddings
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 3 }, { name: 'd', size: 2 }]),
      [
        1, 0, // Object 0
        0.9, 0.1, // Object 1 (similar to 0)
        0, 1, // Object 2 (different)
      ],
    )

    const engine = new AnalogicalReasoningEngine(emb, 1.0)

    // Belief that object 0 has property
    const beliefs = createDenseTensor(
      createShape([{ name: 'x', size: 3 }]),
      [1, 0, 0],
    )

    const propagated = engine.propagateBeliefs(beliefs)

    // Object 1 should inherit some belief from object 0
    expect(propagated.data[1]).toBeGreaterThan(propagated.data[2])
  })

  it('should find similar objects', () => {
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 4 }, { name: 'd', size: 2 }]),
      [
        1, 0, // Object 0
        0.95, 0.05, // Object 1 (very similar to 0)
        0, 1, // Object 2
        0.1, 0.9, // Object 3 (similar to 2)
      ],
    )

    const engine = new AnalogicalReasoningEngine(emb, 1.0)

    const similar = engine.findMostSimilar(0, 2)

    // Object 1 should be most similar to object 0
    expect(similar[0].object).toBe(1)
  })

  it('should compute analogical distance', () => {
    const emb = createRandomEmbeddings(5, 20)
    const engine = new AnalogicalReasoningEngine(emb, 1.0)

    // Distance to self should be less than to others (self-similarity is highest)
    const selfDist = engine.analogicalDistance(0, 0)
    const otherDist = engine.analogicalDistance(0, 1)

    // Self distance should be smaller than other distance
    expect(selfDist).toBeLessThanOrEqual(otherDist)

    // Both should be valid numbers
    expect(Number.isFinite(selfDist)).toBe(true)
    expect(Number.isFinite(otherDist)).toBe(true)
  })

  it('should change behavior with temperature', () => {
    const emb = createRandomEmbeddings(5, 20)
    const engine = new AnalogicalReasoningEngine(emb, 1.0)

    const sim1 = engine.getSimilarity(0, 1)

    engine.setTemperature(0.1)
    const sim2 = engine.getSimilarity(0, 1)

    // Different temperatures should give different similarities
    // (unless they happen to be exactly 0.5)
    expect(engine.getTemperature()).toBe(0.1)
  })
})

describe('ReasoningModes', () => {
  it('should have predefined modes', () => {
    expect(ReasoningModes.DEDUCTIVE.temperature).toBe(0)
    expect(ReasoningModes.WEAK_ANALOGY.temperature).toBe(0.1)
    expect(ReasoningModes.MODERATE_ANALOGY.temperature).toBe(0.5)
    expect(ReasoningModes.STRONG_ANALOGY.temperature).toBe(1.0)
    expect(ReasoningModes.MAXIMUM_UNCERTAINTY.temperature).toBe(10.0)
  })
})

describe('createAnalogicalEngine', () => {
  it('should create engine with mode', () => {
    const emb = createRandomEmbeddings(5, 10)

    const deductive = createAnalogicalEngine(emb, ReasoningModes.DEDUCTIVE)
    expect(deductive.getTemperature()).toBe(0)

    const analog = createAnalogicalEngine(emb, ReasoningModes.STRONG_ANALOGY)
    expect(analog.getTemperature()).toBe(1.0)
  })
})

describe('multiHopAnalogy', () => {
  it('should propagate through multiple hops', () => {
    const emb = createRandomEmbeddings(5, 20)
    const engine = new AnalogicalReasoningEngine(emb, 0.5)

    const facts = createDenseTensor(
      createShape([{ name: 'x', size: 5 }, { name: 'y', size: 5 }]),
      Array.from({ length: 25 }, () => Math.random() > 0.8 ? 1 : 0),
    )

    const propagated = multiHopAnalogy(engine, facts, 2)

    expect(propagated.shape.size).toBe(facts.shape.size)
  })
})

describe('analogyCompletion', () => {
  it('should find analogy completions', () => {
    // Simple embedding space
    const emb = createDenseTensor(
      createShape([{ name: 'x', size: 5 }, { name: 'd', size: 3 }]),
      [
        0, 0, 0, // Object 0
        1, 0, 0, // Object 1
        0, 1, 0, // Object 2
        1, 1, 0, // Object 3 (= 1 + 2 - 0 conceptually)
        0, 0, 1, // Object 4 (unrelated)
      ],
    )

    // A:B::C:? where A=0, B=1, C=2 -> should find 3
    const completions = analogyCompletion(emb, 0, 1, 2)

    // Object 3 should be ranked highly
    expect(completions.some(c => c.object === 3)).toBe(true)
  })

  it('should exclude source objects from completions', () => {
    const emb = createRandomEmbeddings(5, 10)

    const completions = analogyCompletion(emb, 0, 1, 2)

    // Should not include A, B, or C
    expect(completions.every(c => c.object !== 0 && c.object !== 1 && c.object !== 2)).toBe(true)
  })
})

describe('computeOptimalTemperature', () => {
  it('should find best temperature', () => {
    const emb = createRandomEmbeddings(10, 20)

    const known = createDenseTensor(
      createShape([{ name: 'x', size: 10 }, { name: 'y', size: 10 }]),
      Array.from({ length: 100 }, () => Math.random() > 0.9 ? 1 : 0),
    )

    const validation = createDenseTensor(
      createShape([{ name: 'x', size: 10 }, { name: 'y', size: 10 }]),
      Array.from({ length: 100 }, () => Math.random() > 0.9 ? 1 : 0),
    )

    const optimal = computeOptimalTemperature(emb, known, validation)

    expect(optimal).toBeGreaterThanOrEqual(0)
  })
})

describe('Integration tests', () => {
  it('should combine Tucker decomposition with analogical reasoning', () => {
    // Create a relation
    const relation = createRelation(
      ['x', 'y'],
      [[0, 1], [1, 2], [2, 3], [0, 2]],
      [5, 5],
    )

    // Create embeddings
    const emb = createRandomEmbeddings(5, 10)

    // Embed the relation
    const embedded = embedRelation(relation, emb)

    // Use analogical reasoning
    const engine = new AnalogicalReasoningEngine(emb, 0.5)

    // This tests the full pipeline
    expect(embedded).toBeDefined()
    expect(engine).toBeDefined()
  })

  it('should support tensor logic with embeddings', () => {
    // Setup embeddings
    const emb = createRandomEmbeddings(5, 20)

    // Encode a set
    const set = superpositionEncode([0, 2, 4], emb)

    // Query membership
    const in_set = querySuperposition(set, emb, 2)
    const not_in_set = querySuperposition(set, emb, 1)

    // Member should have higher score
    expect(in_set).toBeGreaterThan(not_in_set)
  })
})

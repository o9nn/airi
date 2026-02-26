# @proj-airi/tensor-logic

A TypeScript implementation of [Tensor Logic](https://tensor-logic.org/), an AI programming language that unifies deep learning and symbolic AI through tensor equations and Einstein summation.

Based on the foundational paper: [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) by Pedro Domingos.

## Features

- **Tensor Types**: Dense and sparse tensors with Einstein notation indices
- **Einstein Summation**: Join, projection, Hadamard product, and arithmetic operations
- **Nonlinearities**: Step, sigmoid, ReLU, softmax, GELU, Swish, and more
- **Parser**: Full tensor equation syntax parser (LHS = nonlinearity(RHS))
- **Datalog Support**: Rule parsing for symbolic reasoning
- **Forward Chaining**: Inference engine with fixpoint detection
- **Backward Chaining**: Inference with dependency resolution
- **Automatic Differentiation**: Gradient tape recording for learning
- **Tucker Decomposition**: Embedding-based reasoning
- **Analogical Reasoning**: Temperature-controlled inference (T=0 deductive, T>0 analogical)

## Installation

```shell
pnpm i @proj-airi/tensor-logic
```

## Quick Start

```typescript
import {
  createDenseTensor,
  createShape,
  execute,
  parse,
} from '@proj-airi/tensor-logic'

// Define tensors
const W = createDenseTensor(
  createShape([{ name: 'i', size: 3 }]),
  [1, 2, 3]
)
const X = createDenseTensor(
  createShape([{ name: 'i', size: 3 }]),
  [0.1, 0.2, 0.3]
)

// Define program using tensor logic syntax
const program = parse('Y = sigmoid(W[i] * X[i])')

// Execute the program
const result = execute(program, new Map([['W', W], ['X', X]]))
// Use result.tensors.get('Y') to access the computed tensor
```

## Module Structure

### Core (`@proj-airi/tensor-logic/core`)

Fundamental tensor types and operations:

```typescript
import {
  add,
  // Tensor creation
  createDenseTensor,
  createRelation,
  createShape,
  createSparseTensor,
  // Operations
  einsum,
  eye,
  gelu,
  hadamard,
  join,
  ones,
  project,
  randn,
  random,
  relu,
  scale,
  // Nonlinearities
  sigmoid,
  softmax,
  step,
  subtract,
  swish,
  zeros,
} from '@proj-airi/tensor-logic/core'
```

### Parser (`@proj-airi/tensor-logic/parser`)

Parse tensor logic equations and Datalog rules:

```typescript
import {
  parse,
  parseEquation,
  parseRule,
} from '@proj-airi/tensor-logic/parser'

// Parse a tensor equation
const equation = parseEquation('Z[i,k] = softmax(X[i,j] * Y[j,k])')

// Parse a Datalog rule
const rule = parseRule('ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)')

// Parse a full program
const program = parse(`
  H[i,d] = relu(X[i,j] * W1[j,d])
  Y[i,k] = softmax(H[i,d] * W2[d,k])
`)
```

### Inference (`@proj-airi/tensor-logic/inference`)

Forward and backward chaining engines:

```typescript
import {
  BackwardChainingEngine,
  execute,
  ForwardChainingEngine,
} from '@proj-airi/tensor-logic/inference'

// Forward chaining (bottom-up)
const engine = new ForwardChainingEngine()
engine.addTensor('parent', parentRelation)
engine.addRule('ancestor(X,Y) :- parent(X,Y)')
engine.addRule('ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)')
const result = engine.runToFixpoint()

// Backward chaining (top-down, goal-directed)
const backwardEngine = new BackwardChainingEngine()
backwardEngine.query('ancestor', ['alice', 'charlie'])
```

### Autodiff (`@proj-airi/tensor-logic/autodiff`)

Automatic differentiation for learning:

```typescript
import {
  AdamOptimizer,
  DifferentiableEngine,
  LossFunctions,
} from '@proj-airi/tensor-logic/autodiff'

const engine = new DifferentiableEngine()

// Define parameters
engine.addParameter('W', weightTensor)

// Forward pass
const output = engine.forward(program, inputs)

// Compute loss
const loss = LossFunctions.crossEntropy(output, targets)

// Backward pass
const gradients = engine.backward(loss)

// Update parameters
const optimizer = new AdamOptimizer({ learningRate: 0.001 })
optimizer.step(engine.parameters, gradients)
```

### Embeddings (`@proj-airi/tensor-logic/embeddings`)

Tucker decomposition and analogical reasoning:

```typescript
import {
  AnalogicalReasoningEngine,
  tuckerDecompose,
} from '@proj-airi/tensor-logic/embeddings'

// Decompose a relation into embeddings
const { core, factors } = tuckerDecompose(relationTensor, {
  ranks: [64, 64, 64],
})

// Analogical reasoning with temperature control
const analogyEngine = new AnalogicalReasoningEngine({
  temperature: 0.5, // 0 = deductive, >0 = analogical
})

// Complete analogies: king - man + woman = queen
const result = analogyEngine.complete(['king', 'man', 'woman'])
```

## Key Concepts

### Einstein Summation

Tensor Logic uses Einstein notation where repeated indices imply summation:

```
Z[i,k] = X[i,j] * Y[j,k]  // Matrix multiplication (j is summed)
```

### Tensor Join

Combines tensors by aligning shared indices:

```typescript
const result = join(tensorA, tensorB) // Aligns and multiplies on shared indices
```

### Tensor Projection

Eliminates indices by summing over them:

```typescript
const result = project(tensor, ['j']) // Sum over index j
```

### Forward Chaining

Bottom-up inference that derives all consequences:

```typescript
// Given: parent(alice, bob), parent(bob, charlie)
// Rules: ancestor(X,Y) :- parent(X,Y)
//        ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
// Derives: ancestor(alice, bob), ancestor(bob, charlie), ancestor(alice, charlie)
```

### Temperature-Controlled Reasoning

- **T = 0**: Pure deductive reasoning (Boolean step function)
- **T > 0**: Analogical reasoning with soft matching
- **Higher T**: More exploratory, considers distant analogies

## Examples

### Neural Network Layer

```typescript
const program = parse(`
  H[i,d] = relu(X[i,n] * W1[n,d] + B1[d])
  Y[i,k] = softmax(H[i,d] * W2[d,k] + B2[k])
`)
```

### Knowledge Graph Reasoning

```typescript
const program = parse(`
  friend(X,Y) :- knows(X,Y), knows(Y,X)
  colleague(X,Y) :- worksAt(X,C), worksAt(Y,C), X != Y
  connection(X,Y) :- friend(X,Y)
  connection(X,Y) :- colleague(X,Y)
`)
```

### Attention Mechanism

```typescript
const program = parse(`
  Q[i,d] = X[i,n] * Wq[n,d]
  K[j,d] = X[j,n] * Wk[n,d]
  V[j,v] = X[j,n] * Wv[n,v]
  A[i,j] = softmax(Q[i,d] * K[j,d] / sqrt_dk)
  Y[i,v] = A[i,j] * V[j,v]
`)
```

## API Reference

See the TypeScript types and JSDoc comments in the source code for detailed API documentation.

## Testing

```shell
pnpm test        # Run tests in watch mode
pnpm test:run    # Run tests once
```

## License

[MIT](../../LICENSE)

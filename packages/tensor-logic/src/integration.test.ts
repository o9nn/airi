/**
 * End-to-end integration tests.
 *
 * The central claim of tensor logic is that a single construct — the tensor
 * equation `LHS = [nonlinearity](RHS)` — expresses both neural and symbolic
 * computation. The unit tests cover each primitive in isolation; these tests
 * assert that the paradigms from the paper actually *execute* and produce
 * numerically correct results, rather than merely parsing.
 *
 * @see https://arxiv.org/abs/2510.12269
 */

import { describe, expect, it } from 'vitest'

import { createDenseTensor, createShape } from './core/types'
import { execute } from './inference/engine'

function tensor(dims: Array<[string, number]>, data: number[]) {
  return createDenseTensor(
    createShape(dims.map(([name, size]) => ({ name, size }))),
    data,
  )
}

function read(result: { tensors: Map<string, any> }, name: string) {
  const t = result.tensors.get(name)
  if (!t)
    throw new Error(`tensor '${name}' was not produced by the program`)
  return t
}

describe('einstein summation semantics', () => {
  it('sums out indices absent from the left-hand side', () => {
    // Marg[a] = J[a,b] projects away b — the defining behaviour that makes a
    // tensor equation equivalent to a Datalog rule with an existential.
    const J = tensor([['a', 2], ['b', 3]], [1, 2, 3, 4, 5, 6])

    const result = execute('Marg[a] = J[a,b]', new Map([['J', J]]))
    const marg = read(result, 'Marg')

    expect(Array.from(marg.data)).toEqual([6, 15])
    expect(marg.shape.indices.map((i: any) => i.name)).toEqual(['a'])
  })

  it('contracts shared indices in a product', () => {
    // Dot[.] = A[i] * B[i] contracts i away entirely, leaving a scalar.
    const A = tensor([['i', 3]], [1, 2, 3])
    const B = tensor([['i', 3]], [4, 5, 6])

    const result = execute('Dot = A[i] * B[i]', new Map([['A', A], ['B', B]]))

    expect(read(result, 'Dot').data[0]).toBe(32)
  })
})

describe('neural paradigm', () => {
  it('computes a single-head attention block', () => {
    // Q and K are the 2x2 identity, so Score is the identity and each softmax
    // row is [e/(e+1), 1/(e+1)] = [0.7311, 0.2689].
    const Q = tensor([['p', 2], ['d', 2]], [1, 0, 0, 1])
    const K = tensor([['q', 2], ['d', 2]], [1, 0, 0, 1])
    const V = tensor([['q', 2], ['e', 2]], [10, 20, 30, 40])

    const result = execute(
      `
        Score[p,q] = Q[p,d] * K[q,d]
        Attn[p,q] = softmax(Score[p,q])
        Out[p,e] = Attn[p,q] * V[q,e]
      `,
      new Map([['Q', Q], ['K', K], ['V', V]]),
    )

    expect(Array.from(read(result, 'Score').data)).toEqual([1, 0, 0, 1])

    const attn = read(result, 'Attn')
    expect(attn.data[0]).toBeCloseTo(0.7310585, 6)
    expect(attn.data[1]).toBeCloseTo(0.2689414, 6)
    // Every attention row must be a distribution.
    expect(attn.data[0] + attn.data[1]).toBeCloseTo(1, 10)
    expect(attn.data[2] + attn.data[3]).toBeCloseTo(1, 10)

    // Out = Attn @ V, a convex combination of the value rows.
    const out = read(result, 'Out')
    expect(out.data[0]).toBeCloseTo(15.3788284, 6)
    expect(out.data[1]).toBeCloseTo(25.3788284, 6)
    expect(out.data[2]).toBeCloseTo(24.6211716, 6)
    expect(out.data[3]).toBeCloseTo(34.6211716, 6)
  })

  it('computes an MLP layer with bias and relu', () => {
    // W1 @ X = [1*2 + (-1)*1, 0.5*2 + 0.5*1] = [1, 1.5]; + bias = [1.1, 1.3].
    const W1 = tensor([['h', 2], ['i', 2]], [1, -1, 0.5, 0.5])
    const X = tensor([['i', 2]], [2, 1])
    const B1 = tensor([['h', 2]], [0.1, -0.2])

    const result = execute(
      'Hid[h] = relu(W1[h,i] * X[i] + B1[h])',
      new Map([['W1', W1], ['X', X], ['B1', B1]]),
    )
    const hid = read(result, 'Hid')

    expect(hid.data[0]).toBeCloseTo(1.1, 10)
    expect(hid.data[1]).toBeCloseTo(1.3, 10)
  })

  it('clamps negative pre-activations to zero through relu', () => {
    // W @ X = [-3], so relu must gate it to 0 rather than pass it through.
    const W = tensor([['h', 1], ['i', 1]], [-3])
    const X = tensor([['i', 1]], [1])

    const result = execute('Hid[h] = relu(W[h,i] * X[i])', new Map([['W', W], ['X', X]]))

    expect(read(result, 'Hid').data[0]).toBe(0)
  })
})

describe('probabilistic paradigm', () => {
  it('multiplies factors and marginalises in one equation', () => {
    // P(a) = sum_b P(a|b) P(b) — belief propagation as a single tensor equation.
    const Cond = tensor([['a', 2], ['b', 2]], [0.9, 0.1, 0.2, 0.8])
    const Prior = tensor([['b', 2]], [0.6, 0.4])

    const result = execute(
      'P[a] = Cond[a,b] * Prior[b]',
      new Map([['Cond', Cond], ['Prior', Prior]]),
    )
    const p = read(result, 'P')

    expect(p.data[0]).toBeCloseTo(0.58, 10) // 0.9*0.6 + 0.1*0.4
    expect(p.data[1]).toBeCloseTo(0.44, 10) // 0.2*0.6 + 0.8*0.4
  })
})

describe('symbolic paradigm', () => {
  it('derives a join over boolean relations', () => {
    // Grandparent(x,z) <- Parent(x,y), Parent(y,z), written as a tensor
    // equation over 0/1 tensors. step() restores booleanity after the sum.
    // Chain: 0 -> 1 -> 2, so the only grandparent fact is (0,2).
    const Parent = tensor([['x', 3], ['y', 3]], [
      0,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
    ])

    const result = execute(
      'Grand[x,z] = step(Parent[x,y] * Parent[y,z])',
      new Map([['Parent', Parent]]),
    )
    const grand = read(result, 'Grand')

    expect(Array.from(grand.data)).toEqual([
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
    ])
  })

  it('makes every derived truth value boolean', () => {
    // Two independent derivations of the same fact would sum to 2; step must
    // collapse that back to 1 so the result stays a relation.
    const R = tensor([['x', 2], ['y', 2]], [1, 1, 0, 0])
    const S = tensor([['y', 2], ['z', 2]], [1, 0, 1, 0])

    const result = execute(
      'T[x,z] = step(R[x,y] * S[y,z])',
      new Map([['R', R], ['S', S]]),
    )

    // R row 0 joins S through both y=0 and y=1, giving a raw count of 2.
    expect(Array.from(read(result, 'T').data)).toEqual([1, 0, 0, 0])
  })
})

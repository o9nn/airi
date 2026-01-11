import { describe, expect, it } from 'vitest'

import { getIndexNames, getTensorNames, printExpression, printStatement } from './ast'
import { parse, parseEquation, ParseError, parseRule } from './parser'
import { tokenize, TokenType } from './tokenizer'

describe('Tokenizer', () => {
  it('should tokenize basic equation', () => {
    const tokens = tokenize('Y = W[i] * X[i]')

    expect(tokens.some(t => t.type === TokenType.IDENTIFIER && t.value === 'Y')).toBe(true)
    expect(tokens.some(t => t.type === TokenType.EQUALS)).toBe(true)
    expect(tokens.some(t => t.type === TokenType.LBRACKET)).toBe(true)
    expect(tokens.some(t => t.type === TokenType.RBRACKET)).toBe(true)
    expect(tokens.some(t => t.type === TokenType.STAR)).toBe(true)
    expect(tokens[tokens.length - 1].type).toBe(TokenType.EOF)
  })

  it('should tokenize numbers', () => {
    const tokens = tokenize('X = 3.14 + 2e-5')

    const numbers = tokens.filter(t => t.type === TokenType.NUMBER)
    expect(numbers).toHaveLength(2)
    expect(numbers[0].value).toBe('3.14')
    expect(numbers[1].value).toBe('2e-5')
  })

  it('should tokenize virtual indices', () => {
    const tokens = tokenize('X[*t+1]')

    expect(tokens.some(t => t.type === TokenType.STAR)).toBe(true)
    expect(tokens.some(t => t.type === TokenType.PLUS)).toBe(true)
  })

  it('should tokenize Datalog arrow', () => {
    const tokens = tokenize('A(x,z) <- B(x,y)')

    expect(tokens.some(t => t.type === TokenType.ARROW)).toBe(true)
  })

  it('should skip comments', () => {
    const tokens = tokenize('Y = X # this is a comment\nZ = W')

    expect(tokens.filter(t => t.type === TokenType.IDENTIFIER).map(t => t.value))
      .toEqual(['Y', 'X', 'Z', 'W'])
  })

  it('should handle slicing notation', () => {
    const tokens = tokenize('X[4:8]')

    expect(tokens.some(t => t.type === TokenType.COLON)).toBe(true)
    expect(tokens.filter(t => t.type === TokenType.NUMBER)).toHaveLength(2)
  })
})

describe('Parser - Basic Equations', () => {
  it('should parse simple assignment', () => {
    const eq = parseEquation('Y = X')

    expect(eq.lhs.name).toBe('Y')
    expect(eq.rhs.type).toBe('tensor')
    expect((eq.rhs as any).name).toBe('X')
  })

  it('should parse tensor with indices', () => {
    const eq = parseEquation('Y[i,j] = X[i,j]')

    expect(eq.lhs.indices).toHaveLength(2)
    expect(eq.lhs.indices[0].name).toBe('i')
    expect(eq.lhs.indices[1].name).toBe('j')
  })

  it('should parse virtual indices', () => {
    const eq = parseEquation('X[i,*t+1] = Y[i,*t]')

    const rhsIndices = (eq.rhs as any).indices
    expect(rhsIndices[1].virtual).toBe(true)
    expect(rhsIndices[1].name).toBe('t')
  })

  it('should parse index with offset', () => {
    const eq = parseEquation('X[t+1] = Y[t]')

    expect(eq.lhs.indices[0].offset).toBe(1)
  })

  it('should parse index with stride', () => {
    const eq = parseEquation('Pooled[x/2] = X[x]')

    expect(eq.lhs.indices[0].stride).toBe(2)
  })
})

describe('Parser - Expressions', () => {
  it('should parse addition', () => {
    const eq = parseEquation('Y = A + B')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('+')
  })

  it('should parse subtraction', () => {
    const eq = parseEquation('Y = A - B')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('-')
  })

  it('should parse multiplication', () => {
    const eq = parseEquation('Y = A * B')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('*')
  })

  it('should parse explicit multiplication', () => {
    // Note: Implicit multiplication (W[i,j] X[j]) is not supported due to
    // ambiguity with statement boundaries. Use explicit * operator.
    const eq = parseEquation('Y = W[i,j] * X[j]')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('*')
  })

  it('should parse division', () => {
    const eq = parseEquation('Y = A / B')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('/')
  })

  it('should parse exponentiation', () => {
    const eq = parseEquation('Y = X^2')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('^')
  })

  it('should parse negation', () => {
    const eq = parseEquation('Y = -X')

    expect(eq.rhs.type).toBe('unary')
    expect((eq.rhs as any).operator).toBe('-')
  })

  it('should parse numeric literals', () => {
    const eq = parseEquation('Y = 3.14')

    expect(eq.rhs.type).toBe('number')
    expect((eq.rhs as any).value).toBeCloseTo(3.14)
  })

  it('should respect operator precedence', () => {
    const eq = parseEquation('Y = A + B * C')

    // Should be A + (B * C)
    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('+')
    expect((eq.rhs as any).right.type).toBe('binary')
    expect((eq.rhs as any).right.operator).toBe('*')
  })

  it('should parse parentheses', () => {
    const eq = parseEquation('Y = (A + B) * C')

    expect(eq.rhs.type).toBe('binary')
    expect((eq.rhs as any).operator).toBe('*')
    expect((eq.rhs as any).left.type).toBe('binary')
    expect((eq.rhs as any).left.operator).toBe('+')
  })
})

describe('Parser - Function Calls', () => {
  it('should parse function with single argument', () => {
    const eq = parseEquation('Y = sigmoid(X)')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('sigmoid')
    expect((eq.rhs as any).arguments).toHaveLength(1)
  })

  it('should parse function with multiple arguments', () => {
    const eq = parseEquation('Y = clamp(X, 0, 1)')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('clamp')
    expect((eq.rhs as any).arguments).toHaveLength(3)
  })

  it('should parse nested function calls', () => {
    const eq = parseEquation('Y = sigmoid(relu(X))')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).arguments[0].type).toBe('function')
  })

  it('should parse step function', () => {
    const eq = parseEquation('Y = step(W[i] * X[i])')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('step')
  })

  it('should parse Heaviside shorthand', () => {
    const eq = parseEquation('Y = H(X)')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('H')
  })

  it('should parse softmax', () => {
    const eq = parseEquation('Y = softmax(X)')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('softmax')
  })

  it('should parse lnorm', () => {
    const eq = parseEquation('Y = lnorm(X)')

    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('lnorm')
  })
})

describe('Parser - Datalog Rules', () => {
  it('should parse simple rule', () => {
    const rule = parseRule('Aunt(x,z) <- Sister(x,y), Parent(y,z)')

    expect(rule.head.name).toBe('Aunt')
    expect(rule.head.arguments).toEqual(['x', 'z'])
    expect(rule.body).toHaveLength(2)
    expect(rule.body[0].name).toBe('Sister')
    expect(rule.body[1].name).toBe('Parent')
  })

  it('should parse single body rule', () => {
    const rule = parseRule('Ancestor(x,y) <- Parent(x,y)')

    expect(rule.body).toHaveLength(1)
  })

  it('should parse recursive rule', () => {
    const rule = parseRule('Ancestor(x,z) <- Ancestor(x,y), Parent(y,z)')

    expect(rule.head.name).toBe('Ancestor')
    expect(rule.body[0].name).toBe('Ancestor')
  })
})

describe('Parser - Full Programs', () => {
  it('should parse multiple equations', () => {
    const program = parse(`
      X[i,1] = sigmoid(W[i,j] * X[j,0])
      X[i,2] = sigmoid(W[i,j] * X[j,1])
      Y = softmax(X[i,2])
    `)

    expect(program.statements).toHaveLength(3)
    expect(program.statements.every(s => s.type === 'equation')).toBe(true)
  })

  it('should parse transformer-like program', () => {
    const program = parse(`
      Query[h,p,d] = W_Q[h,d,e] * Stream[p,e]
      Key[h,p,d] = W_K[h,d,e] * Stream[p,e]
      Value[h,p,d] = W_V[h,d,e] * Stream[p,e]
      Attn[h,p,q] = softmax(Query[h,p,d] * Key[h,q,d])
      Out[h,p,d] = Attn[h,p,q] * Value[h,q,d]
    `)

    expect(program.statements).toHaveLength(5)
  })

  it('should handle comments in program', () => {
    const program = parse(`
      # This computes a simple MLP
      H = relu(W1[i,j] * X[j])  # Hidden layer
      Y = sigmoid(W2[k,i] * H[i])  # Output
    `)

    expect(program.statements).toHaveLength(2)
  })
})

describe('AST Utilities', () => {
  it('should get tensor names from expression', () => {
    const eq = parseEquation('Y = W[i,j] * X[j] + B[i]')
    const names = getTensorNames(eq.rhs)

    expect(names).toContain('W')
    expect(names).toContain('X')
    expect(names).toContain('B')
    expect(names).not.toContain('Y')
  })

  it('should get index names from expression', () => {
    const eq = parseEquation('Y[i,k] = W[i,j] * X[j,k]')
    const names = getIndexNames(eq.rhs)

    expect(names).toContain('i')
    expect(names).toContain('j')
    expect(names).toContain('k')
  })

  it('should print expression', () => {
    const eq = parseEquation('Y = sigmoid(W[i,j] * X[j])')
    const printed = printExpression(eq.rhs)

    expect(printed).toContain('sigmoid')
    expect(printed).toContain('W')
    expect(printed).toContain('X')
  })

  it('should print statement', () => {
    const eq = parseEquation('Y[i] = relu(X[i])')
    const printed = printStatement(eq)

    expect(printed).toContain('Y[i]')
    expect(printed).toContain('=')
    expect(printed).toContain('relu')
  })
})

describe('Parser Error Handling', () => {
  it('should throw on missing equals', () => {
    expect(() => parseEquation('Y X')).toThrow(ParseError)
  })

  it('should throw on unclosed bracket', () => {
    expect(() => parseEquation('Y = X[i')).toThrow(ParseError)
  })

  it('should throw on unclosed parenthesis', () => {
    expect(() => parseEquation('Y = sigmoid(X')).toThrow(ParseError)
  })

  it('should provide location in error', () => {
    try {
      parseEquation('Y = X[')
    }
    catch (e) {
      expect(e).toBeInstanceOf(ParseError)
      expect((e as ParseError).line).toBeGreaterThan(0)
      expect((e as ParseError).column).toBeGreaterThan(0)
    }
  })
})

describe('Complex Examples', () => {
  it('should parse attention mechanism', () => {
    const eq = parseEquation('Attn[b,h,p,d] = softmax(Q[b,h,p,k] * K[b,h,q,k]) * V[b,h,q,d]')

    expect(eq.lhs.indices).toHaveLength(4)
    expect(eq.rhs.type).toBe('binary')
  })

  it('should parse convolution-like expression', () => {
    // Note: The standard tensor notation uses explicit indices for convolution
    // where the filter slides over spatial dimensions via Einstein summation
    const eq = parseEquation('Features[x,y,c] = relu(Filter[dx,dy,cin,c] * Image[x,y,cin])')

    // This should parse the function call with binary expression inside
    expect(eq.rhs.type).toBe('function')
    expect((eq.rhs as any).name).toBe('relu')
    const funcArg = (eq.rhs as any).arguments[0]
    expect(funcArg.type).toBe('binary')
  })

  it('should parse pooling-like expression', () => {
    const eq = parseEquation('Pooled[x/2,y/2] = Features[x,y]')

    expect(eq.lhs.indices[0].stride).toBe(2)
    expect(eq.lhs.indices[1].stride).toBe(2)
  })

  it('should parse RNN-like expression', () => {
    const eq = parseEquation('H[*t+1,d] = sigmoid(W[d,e] * H[*t,e] + V[d,f] * X[t,f])')

    expect(eq.lhs.indices[0].virtual).toBe(true)
    expect(eq.lhs.indices[0].offset).toBe(1)
  })

  it('should parse loss function', () => {
    const eq = parseEquation('Loss = (Y[e] - Target[e])^2')

    expect((eq.rhs as any).operator).toBe('^')
    expect((eq.rhs as any).left.operator).toBe('-')
  })
})

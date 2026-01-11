/**
 * Tensor Logic Parser
 *
 * Parses tensor logic equations from tokens into AST
 *
 * Grammar:
 *   program     → statement*
 *   statement   → equation | rule
 *   equation    → tensorRef '=' expression
 *   rule        → relation '←' relation (',' relation)*
 *   expression  → term (('+' | '-') term)*
 *   term        → factor (('*' | '/') factor)*
 *   factor      → unary ('^' factor)?
 *   unary       → '-' unary | primary
 *   primary     → number | functionCall | tensorRef | relation | '(' expression ')'
 *   tensorRef   → IDENTIFIER ('[' indexList ']')?
 *   indexList   → indexRef (',' indexRef)*
 *   indexRef    → '*'? IDENTIFIER (('+' | '-') NUMBER)? ('/' NUMBER)? | NUMBER ':' NUMBER
 *   functionCall → IDENTIFIER '(' argumentList ')'
 *   argumentList → expression (',' expression)*
 *   relation    → IDENTIFIER '(' varList ')'
 *   varList     → IDENTIFIER (',' IDENTIFIER)*
 */

import type {
  BinaryOp,
  DatalogRule,
  Expression,
  FunctionCall,
  IndexRef,
  NumberLiteral,
  Program,
  Relation,
  Statement,
  TensorEquation,
  TensorRef,
  UnaryOp,
} from './ast'
import type { Token } from './tokenizer'
import { TokenType, tokenize } from './tokenizer'

/**
 * Parse error with location information
 */
export class ParseError extends Error {
  constructor(
    message: string,
    public token: Token,
    public line: number,
    public column: number,
  ) {
    super(`Parse error at line ${line}, column ${column}: ${message}`)
    this.name = 'ParseError'
  }
}

/**
 * Parser for tensor logic programs
 */
export class Parser {
  private tokens: Token[]
  private position = 0

  constructor(tokens: Token[]) {
    this.tokens = tokens
  }

  /**
   * Parse a complete program
   */
  parse(): Program {
    const statements: Statement[] = []

    while (!this.isAtEnd()) {
      const stmt = this.parseStatement()
      if (stmt) {
        statements.push(stmt)
      }
    }

    return {
      type: 'program',
      statements,
    }
  }

  /**
   * Parse a single statement
   */
  private parseStatement(): Statement | null {
    // Look ahead to determine statement type
    const first = this.current()

    if (first.type !== TokenType.IDENTIFIER) {
      return null
    }

    // Check if this is a Datalog rule (has parentheses after name and then <-)
    const saved = this.position
    this.advance()

    if (this.check(TokenType.LPAREN)) {
      // Could be relation or function call
      // Look for arrow after the parentheses
      let parenDepth = 0
      let pos = this.position

      while (pos < this.tokens.length) {
        const t = this.tokens[pos]
        if (t.type === TokenType.LPAREN)
          parenDepth++
        else if (t.type === TokenType.RPAREN)
          parenDepth--
        else if (t.type === TokenType.ARROW && parenDepth === 0) {
          // This is a Datalog rule
          this.position = saved
          return this.parseRule()
        }
        else if (t.type === TokenType.EQUALS && parenDepth === 0) {
          // This is an equation
          break
        }
        pos++
      }
    }

    // Reset and parse as equation
    this.position = saved
    return this.parseEquation()
  }

  /**
   * Parse a tensor equation: LHS = RHS
   */
  private parseEquation(): TensorEquation {
    const lhs = this.parseTensorRef()

    // Check for normalization indices (marked with .)
    let normalizeIndices: string[] | undefined

    // Look for dot notation in the LHS indices for normalization
    for (const idx of lhs.indices) {
      // Indices ending with . indicate normalization target
      if (idx.name.endsWith('.')) {
        if (!normalizeIndices)
          normalizeIndices = []
        normalizeIndices.push(idx.name.slice(0, -1))
        idx.name = idx.name.slice(0, -1)
      }
    }

    this.consume(TokenType.EQUALS, 'Expected \'=\' after left-hand side')

    const rhs = this.parseExpression()

    return {
      type: 'equation',
      lhs,
      rhs,
      normalizeIndices,
    }
  }

  /**
   * Parse a Datalog rule: Head <- Body1, Body2, ...
   */
  private parseRule(): DatalogRule {
    const head = this.parseRelation()

    this.consume(TokenType.ARROW, 'Expected \'<-\' in Datalog rule')

    const body: Relation[] = [this.parseRelation()]

    while (this.check(TokenType.COMMA)) {
      this.advance()
      body.push(this.parseRelation())
    }

    return {
      type: 'rule',
      head,
      body,
    }
  }

  /**
   * Parse an expression (lowest precedence: addition/subtraction)
   */
  private parseExpression(): Expression {
    let left = this.parseTerm()

    while (this.check(TokenType.PLUS) || this.check(TokenType.MINUS)) {
      const operator = this.advance().value as '+' | '-'
      const right = this.parseTerm()
      left = {
        type: 'binary',
        operator,
        left,
        right,
      } as BinaryOp
    }

    return left
  }

  /**
   * Parse a term (multiplication/division)
   */
  private parseTerm(): Expression {
    let left = this.parseFactor()

    while (this.check(TokenType.STAR) || this.check(TokenType.SLASH)) {
      const operator = this.advance().value as '*' | '/'
      const right = this.parseFactor()
      left = {
        type: 'binary',
        operator,
        left,
        right,
      } as BinaryOp
    }

    // Note: Implicit multiplication (two tensor refs adjacent like "A[i] B[j]")
    // is intentionally NOT supported because it creates ambiguity with statement
    // boundaries in multi-statement programs. Use explicit * operator instead.

    return left
  }

  /**
   * Parse a factor (exponentiation)
   */
  private parseFactor(): Expression {
    const base = this.parseUnary()

    if (this.check(TokenType.CARET)) {
      this.advance()
      const exponent = this.parseFactor() // Right associative
      return {
        type: 'binary',
        operator: '^',
        left: base,
        right: exponent,
      } as BinaryOp
    }

    return base
  }

  /**
   * Parse a unary expression
   */
  private parseUnary(): Expression {
    if (this.check(TokenType.MINUS)) {
      this.advance()
      const operand = this.parseUnary()
      return {
        type: 'unary',
        operator: '-',
        operand,
      } as UnaryOp
    }

    return this.parsePrimary()
  }

  /**
   * Parse a primary expression
   */
  private parsePrimary(): Expression {
    // Number literal
    if (this.check(TokenType.NUMBER)) {
      const token = this.advance()
      return {
        type: 'number',
        value: Number.parseFloat(token.value),
      } as NumberLiteral
    }

    // Parenthesized expression
    if (this.check(TokenType.LPAREN)) {
      this.advance()
      const expr = this.parseExpression()
      this.consume(TokenType.RPAREN, 'Expected \')\' after expression')
      return expr
    }

    // Identifier - could be tensor ref, function call, or relation
    if (this.check(TokenType.IDENTIFIER)) {
      const name = this.current().value
      const nextPos = this.position + 1
      const next = nextPos < this.tokens.length ? this.tokens[nextPos] : null

      // Function call
      if (next?.type === TokenType.LPAREN) {
        return this.parseFunctionCall()
      }

      // Tensor reference (with or without indices)
      return this.parseTensorRef()
    }

    throw this.error('Expected expression', this.current())
  }

  /**
   * Parse a tensor reference: Name[i,j,k]
   */
  private parseTensorRef(): TensorRef {
    const name = this.consume(TokenType.IDENTIFIER, 'Expected tensor name').value
    const indices: IndexRef[] = []

    if (this.check(TokenType.LBRACKET)) {
      this.advance()

      if (!this.check(TokenType.RBRACKET)) {
        indices.push(this.parseIndexRef())

        while (this.check(TokenType.COMMA)) {
          this.advance()
          indices.push(this.parseIndexRef())
        }
      }

      this.consume(TokenType.RBRACKET, 'Expected \']\' after indices')
    }

    return {
      type: 'tensor',
      name,
      indices,
    }
  }

  /**
   * Parse an index reference: i, *i, i+1, i/2, 0:10
   */
  private parseIndexRef(): IndexRef {
    // Check for slice notation: number:number
    if (this.check(TokenType.NUMBER)) {
      const startToken = this.advance()
      if (this.check(TokenType.COLON)) {
        this.advance()
        const endToken = this.consume(TokenType.NUMBER, 'Expected end of slice')
        return {
          type: 'index',
          name: '_slice_',
          slice: [Number.parseInt(startToken.value), Number.parseInt(endToken.value)],
        }
      }
      // Just a number as index (unusual but allowed)
      return {
        type: 'index',
        name: startToken.value,
      }
    }

    // Check for virtual index prefix
    let virtual = false
    if (this.check(TokenType.STAR)) {
      this.advance()
      virtual = true
    }

    let name = this.consume(TokenType.IDENTIFIER, 'Expected index name').value

    // Check for normalization marker (dot at end)
    if (this.check(TokenType.DOT)) {
      this.advance()
      name += '.'
    }

    let offset: number | undefined
    let stride: number | undefined

    // Check for offset: +N or -N
    if (this.check(TokenType.PLUS)) {
      this.advance()
      const num = this.consume(TokenType.NUMBER, 'Expected offset value')
      offset = Number.parseInt(num.value)
    }
    else if (this.check(TokenType.MINUS)) {
      this.advance()
      const num = this.consume(TokenType.NUMBER, 'Expected offset value')
      offset = -Number.parseInt(num.value)
    }

    // Check for stride/division: /N
    if (this.check(TokenType.SLASH)) {
      this.advance()
      const num = this.consume(TokenType.NUMBER, 'Expected stride value')
      stride = Number.parseInt(num.value)
    }

    return {
      type: 'index',
      name,
      virtual,
      offset,
      stride,
    }
  }

  /**
   * Parse a function call: name(arg1, arg2, ...)
   */
  private parseFunctionCall(): FunctionCall {
    const name = this.consume(TokenType.IDENTIFIER, 'Expected function name').value
    this.consume(TokenType.LPAREN, 'Expected \'(\' after function name')

    const args: Expression[] = []

    if (!this.check(TokenType.RPAREN)) {
      args.push(this.parseExpression())

      while (this.check(TokenType.COMMA)) {
        this.advance()
        args.push(this.parseExpression())
      }
    }

    this.consume(TokenType.RPAREN, 'Expected \')\' after arguments')

    return {
      type: 'function',
      name,
      arguments: args,
    }
  }

  /**
   * Parse a Datalog relation: Name(x, y, z)
   */
  private parseRelation(): Relation {
    const name = this.consume(TokenType.IDENTIFIER, 'Expected relation name').value
    this.consume(TokenType.LPAREN, 'Expected \'(\' after relation name')

    const args: string[] = []

    if (!this.check(TokenType.RPAREN)) {
      args.push(this.consume(TokenType.IDENTIFIER, 'Expected variable name').value)

      while (this.check(TokenType.COMMA)) {
        this.advance()
        args.push(this.consume(TokenType.IDENTIFIER, 'Expected variable name').value)
      }
    }

    this.consume(TokenType.RPAREN, 'Expected \')\' after relation arguments')

    return {
      type: 'relation',
      name,
      arguments: args,
    }
  }

  // Helper methods

  private current(): Token {
    return this.tokens[this.position]
  }

  private advance(): Token {
    if (!this.isAtEnd()) {
      this.position++
    }
    return this.tokens[this.position - 1]
  }

  private check(type: TokenType): boolean {
    return !this.isAtEnd() && this.current().type === type
  }

  private consume(type: TokenType, message: string): Token {
    if (this.check(type)) {
      return this.advance()
    }
    throw this.error(message, this.current())
  }

  private isAtEnd(): boolean {
    return this.current().type === TokenType.EOF
  }

  private error(message: string, token: Token): ParseError {
    return new ParseError(message, token, token.line, token.column)
  }
}

/**
 * Parse a tensor logic program from source code
 */
export function parse(source: string): Program {
  const tokens = tokenize(source)
  const parser = new Parser(tokens)
  return parser.parse()
}

/**
 * Parse a single tensor equation
 */
export function parseEquation(source: string): TensorEquation {
  const program = parse(source)
  if (program.statements.length !== 1 || program.statements[0].type !== 'equation') {
    throw new Error('Expected exactly one tensor equation')
  }
  return program.statements[0] as TensorEquation
}

/**
 * Parse a Datalog rule
 */
export function parseRule(source: string): DatalogRule {
  const program = parse(source)
  if (program.statements.length !== 1 || program.statements[0].type !== 'rule') {
    throw new Error('Expected exactly one Datalog rule')
  }
  return program.statements[0] as DatalogRule
}

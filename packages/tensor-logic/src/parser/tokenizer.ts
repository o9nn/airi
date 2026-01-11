/**
 * Tensor Logic Tokenizer
 *
 * Tokenizes tensor logic equations for parsing
 */

export enum TokenType {
  IDENTIFIER = 'IDENTIFIER', // Variable/tensor names
  NUMBER = 'NUMBER', // Numeric literals
  LBRACKET = 'LBRACKET', // [
  RBRACKET = 'RBRACKET', // ]
  LPAREN = 'LPAREN', // (
  RPAREN = 'RPAREN', // )
  EQUALS = 'EQUALS', // =
  COMMA = 'COMMA', // ,
  STAR = 'STAR', // * (multiplication or virtual index prefix)
  PLUS = 'PLUS', // +
  MINUS = 'MINUS', // -
  SLASH = 'SLASH', // / (division)
  COLON = 'COLON', // : (slicing)
  DOT = 'DOT', // . (normalization marker)
  CARET = 'CARET', // ^ (power)
  ARROW = 'ARROW', // <- (Datalog rule)
  EOF = 'EOF',
}

export interface Token {
  type: TokenType
  value: string
  position: number
  line: number
  column: number
}

/**
 * Tokenizer for tensor logic equations
 */
export class Tokenizer {
  private input: string
  private position = 0
  private line = 1
  private column = 1

  constructor(input: string) {
    this.input = input
  }

  /**
   * Tokenize the entire input
   */
  tokenize(): Token[] {
    const tokens: Token[] = []

    while (this.position < this.input.length) {
      const token = this.nextToken()
      if (token) {
        tokens.push(token)
      }
    }

    tokens.push({
      type: TokenType.EOF,
      value: '',
      position: this.position,
      line: this.line,
      column: this.column,
    })

    return tokens
  }

  private nextToken(): Token | null {
    this.skipWhitespace()

    if (this.position >= this.input.length) {
      return null
    }

    const startPosition = this.position
    const startLine = this.line
    const startColumn = this.column
    const char = this.input[this.position]

    // Check for two-character tokens first
    if (char === '<' && this.peek(1) === '-') {
      this.advance(2)
      return {
        type: TokenType.ARROW,
        value: '<-',
        position: startPosition,
        line: startLine,
        column: startColumn,
      }
    }

    // Single character tokens
    const singleCharTokens: Record<string, TokenType> = {
      '[': TokenType.LBRACKET,
      ']': TokenType.RBRACKET,
      '(': TokenType.LPAREN,
      ')': TokenType.RPAREN,
      '=': TokenType.EQUALS,
      ',': TokenType.COMMA,
      '*': TokenType.STAR,
      '+': TokenType.PLUS,
      '-': TokenType.MINUS,
      '/': TokenType.SLASH,
      ':': TokenType.COLON,
      '.': TokenType.DOT,
      '^': TokenType.CARET,
    }

    if (char in singleCharTokens) {
      this.advance()
      return {
        type: singleCharTokens[char],
        value: char,
        position: startPosition,
        line: startLine,
        column: startColumn,
      }
    }

    // Numbers
    if (this.isDigit(char) || (char === '.' && this.isDigit(this.peek(1)))) {
      return this.readNumber(startPosition, startLine, startColumn)
    }

    // Identifiers (variable names, function names)
    if (this.isAlpha(char) || char === '_') {
      return this.readIdentifier(startPosition, startLine, startColumn)
    }

    // Unknown character - skip it
    this.advance()
    return null
  }

  private readNumber(startPosition: number, startLine: number, startColumn: number): Token {
    let value = ''
    let hasDot = false
    let hasExponent = false

    while (this.position < this.input.length) {
      const char = this.input[this.position]

      if (this.isDigit(char)) {
        value += char
        this.advance()
      }
      else if (char === '.' && !hasDot && !hasExponent) {
        hasDot = true
        value += char
        this.advance()
      }
      else if ((char === 'e' || char === 'E') && !hasExponent) {
        hasExponent = true
        value += char
        this.advance()
        // Handle optional sign after exponent
        if (this.position < this.input.length) {
          const next = this.input[this.position]
          if (next === '+' || next === '-') {
            value += next
            this.advance()
          }
        }
      }
      else {
        break
      }
    }

    return {
      type: TokenType.NUMBER,
      value,
      position: startPosition,
      line: startLine,
      column: startColumn,
    }
  }

  private readIdentifier(startPosition: number, startLine: number, startColumn: number): Token {
    let value = ''

    while (this.position < this.input.length) {
      const char = this.input[this.position]
      if (this.isAlphaNumeric(char) || char === '_') {
        value += char
        this.advance()
      }
      else {
        break
      }
    }

    return {
      type: TokenType.IDENTIFIER,
      value,
      position: startPosition,
      line: startLine,
      column: startColumn,
    }
  }

  private skipWhitespace(): void {
    while (this.position < this.input.length) {
      const char = this.input[this.position]
      if (char === ' ' || char === '\t' || char === '\r') {
        this.advance()
      }
      else if (char === '\n') {
        this.advance()
        this.line++
        this.column = 1
      }
      else if (char === '#' || (char === '/' && this.peek(1) === '/')) {
        // Skip comment until end of line
        while (this.position < this.input.length && this.input[this.position] !== '\n') {
          this.advance()
        }
      }
      else {
        break
      }
    }
  }

  private advance(count = 1): void {
    for (let i = 0; i < count; i++) {
      if (this.position < this.input.length) {
        if (this.input[this.position] === '\n') {
          this.line++
          this.column = 1
        }
        else {
          this.column++
        }
        this.position++
      }
    }
  }

  private peek(offset = 0): string {
    const pos = this.position + offset
    return pos < this.input.length ? this.input[pos] : ''
  }

  private isDigit(char: string): boolean {
    return char >= '0' && char <= '9'
  }

  private isAlpha(char: string): boolean {
    return (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z')
  }

  private isAlphaNumeric(char: string): boolean {
    return this.isAlpha(char) || this.isDigit(char)
  }
}

/**
 * Tokenize a tensor logic equation string
 */
export function tokenize(input: string): Token[] {
  const tokenizer = new Tokenizer(input)
  return tokenizer.tokenize()
}

/**
 * Tensor Logic Abstract Syntax Tree
 *
 * Defines the AST node types for tensor logic programs
 */

/**
 * Index reference in tensor notation
 */
export interface IndexRef {
  type: 'index'
  /** Index name (e.g., 'i', 'x', 't') */
  name: string
  /** Whether this is a virtual index (prefixed with *) */
  virtual?: boolean
  /** Offset expression (e.g., t+1) */
  offset?: number
  /** Stride/division (e.g., x/S) */
  stride?: number
  /** Slice range [start, end) */
  slice?: [number, number]
}

/**
 * Tensor reference: Name[indices]
 */
export interface TensorRef {
  type: 'tensor'
  /** Tensor name */
  name: string
  /** Index references */
  indices: IndexRef[]
}

/**
 * Numeric constant
 */
export interface NumberLiteral {
  type: 'number'
  value: number
}

/**
 * Binary operation: left op right
 */
export interface BinaryOp {
  type: 'binary'
  operator: '+' | '-' | '*' | '/' | '^'
  left: Expression
  right: Expression
}

/**
 * Unary operation: op expr
 */
export interface UnaryOp {
  type: 'unary'
  operator: '-'
  operand: Expression
}

/**
 * Function call: name(args)
 */
export interface FunctionCall {
  type: 'function'
  name: string
  arguments: Expression[]
}

/**
 * Datalog-style relation: Name(args)
 */
export interface Relation {
  type: 'relation'
  name: string
  arguments: string[]
}

/**
 * Expression union type
 */
export type Expression =
  | TensorRef
  | NumberLiteral
  | BinaryOp
  | UnaryOp
  | FunctionCall
  | Relation

/**
 * Tensor equation: LHS = RHS or LHS = nonlinearity(RHS)
 */
export interface TensorEquation {
  type: 'equation'
  /** Left-hand side tensor reference */
  lhs: TensorRef
  /** Right-hand side expression */
  rhs: Expression
  /** Normalization indices (for softmax/lnorm, marked with .) */
  normalizeIndices?: string[]
}

/**
 * Datalog rule: Head <- Body1, Body2, ...
 */
export interface DatalogRule {
  type: 'rule'
  /** Head relation */
  head: Relation
  /** Body relations (conjunction) */
  body: Relation[]
}

/**
 * Statement union type
 */
export type Statement = TensorEquation | DatalogRule

/**
 * Complete tensor logic program
 */
export interface Program {
  type: 'program'
  statements: Statement[]
}

/**
 * Visitor interface for AST traversal
 */
export interface ASTVisitor<T> {
  visitProgram?(node: Program): T
  visitEquation?(node: TensorEquation): T
  visitRule?(node: DatalogRule): T
  visitTensorRef?(node: TensorRef): T
  visitNumberLiteral?(node: NumberLiteral): T
  visitBinaryOp?(node: BinaryOp): T
  visitUnaryOp?(node: UnaryOp): T
  visitFunctionCall?(node: FunctionCall): T
  visitRelation?(node: Relation): T
  visitIndexRef?(node: IndexRef): T
}

/**
 * Walk the AST and call visitor methods
 */
export function walkAST<T>(node: Expression | Statement | Program, visitor: ASTVisitor<T>): T | undefined {
  if (node.type === 'program') {
    return visitor.visitProgram?.(node)
  }
  if (node.type === 'equation') {
    return visitor.visitEquation?.(node)
  }
  if (node.type === 'rule') {
    return visitor.visitRule?.(node)
  }
  if (node.type === 'tensor') {
    return visitor.visitTensorRef?.(node)
  }
  if (node.type === 'number') {
    return visitor.visitNumberLiteral?.(node)
  }
  if (node.type === 'binary') {
    return visitor.visitBinaryOp?.(node)
  }
  if (node.type === 'unary') {
    return visitor.visitUnaryOp?.(node)
  }
  if (node.type === 'function') {
    return visitor.visitFunctionCall?.(node)
  }
  if (node.type === 'relation') {
    return visitor.visitRelation?.(node)
  }
  return undefined
}

/**
 * Get all tensor names referenced in an expression
 */
export function getTensorNames(expr: Expression): string[] {
  const names: string[] = []

  function collect(e: Expression): void {
    switch (e.type) {
      case 'tensor':
        if (!names.includes(e.name)) {
          names.push(e.name)
        }
        break
      case 'binary':
        collect(e.left)
        collect(e.right)
        break
      case 'unary':
        collect(e.operand)
        break
      case 'function':
        for (const arg of e.arguments) {
          collect(arg)
        }
        break
      case 'relation':
        if (!names.includes(e.name)) {
          names.push(e.name)
        }
        break
    }
  }

  collect(expr)
  return names
}

/**
 * Get all index names used in an expression
 */
export function getIndexNames(expr: Expression): string[] {
  const names: string[] = []

  function collect(e: Expression): void {
    switch (e.type) {
      case 'tensor':
        for (const idx of e.indices) {
          if (!names.includes(idx.name)) {
            names.push(idx.name)
          }
        }
        break
      case 'binary':
        collect(e.left)
        collect(e.right)
        break
      case 'unary':
        collect(e.operand)
        break
      case 'function':
        for (const arg of e.arguments) {
          collect(arg)
        }
        break
    }
  }

  collect(expr)
  return names
}

/**
 * Check if two expressions are structurally equal
 */
export function expressionsEqual(a: Expression, b: Expression): boolean {
  if (a.type !== b.type)
    return false

  switch (a.type) {
    case 'number':
      return a.value === (b as NumberLiteral).value
    case 'tensor': {
      const bt = b as TensorRef
      if (a.name !== bt.name || a.indices.length !== bt.indices.length)
        return false
      return a.indices.every((idx, i) =>
        idx.name === bt.indices[i].name
        && idx.virtual === bt.indices[i].virtual
        && idx.offset === bt.indices[i].offset,
      )
    }
    case 'binary': {
      const bb = b as BinaryOp
      return a.operator === bb.operator
        && expressionsEqual(a.left, bb.left)
        && expressionsEqual(a.right, bb.right)
    }
    case 'unary': {
      const bu = b as UnaryOp
      return a.operator === bu.operator && expressionsEqual(a.operand, bu.operand)
    }
    case 'function': {
      const bf = b as FunctionCall
      if (a.name !== bf.name || a.arguments.length !== bf.arguments.length)
        return false
      return a.arguments.every((arg, i) => expressionsEqual(arg, bf.arguments[i]))
    }
    case 'relation': {
      const br = b as Relation
      if (a.name !== br.name || a.arguments.length !== br.arguments.length)
        return false
      return a.arguments.every((arg, i) => arg === br.arguments[i])
    }
    default:
      return false
  }
}

/**
 * Pretty print an expression
 */
export function printExpression(expr: Expression): string {
  switch (expr.type) {
    case 'number':
      return String(expr.value)
    case 'tensor': {
      if (expr.indices.length === 0)
        return expr.name
      const indices = expr.indices.map((idx) => {
        let s = idx.virtual ? `*${idx.name}` : idx.name
        if (idx.offset !== undefined) {
          s += idx.offset >= 0 ? `+${idx.offset}` : String(idx.offset)
        }
        if (idx.stride !== undefined) {
          s += `/${idx.stride}`
        }
        if (idx.slice !== undefined) {
          s = `${idx.slice[0]}:${idx.slice[1]}`
        }
        return s
      })
      return `${expr.name}[${indices.join(',')}]`
    }
    case 'binary':
      return `(${printExpression(expr.left)} ${expr.operator} ${printExpression(expr.right)})`
    case 'unary':
      return `${expr.operator}${printExpression(expr.operand)}`
    case 'function':
      return `${expr.name}(${expr.arguments.map(printExpression).join(', ')})`
    case 'relation':
      return `${expr.name}(${expr.arguments.join(', ')})`
    default:
      return ''
  }
}

/**
 * Pretty print a statement
 */
export function printStatement(stmt: Statement): string {
  if (stmt.type === 'equation') {
    return `${printExpression(stmt.lhs)} = ${printExpression(stmt.rhs)}`
  }
  else {
    const head = printExpression(stmt.head)
    const body = stmt.body.map(printExpression).join(', ')
    return `${head} <- ${body}`
  }
}

/**
 * Pretty print a program
 */
export function printProgram(program: Program): string {
  return program.statements.map(printStatement).join('\n')
}

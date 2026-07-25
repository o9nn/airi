/**
 * Tensor Logic Parser Module
 *
 * Exports the tokenizer, AST types, and parser
 */

// AST
export type {
  ASTVisitor,
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
export {
  expressionsEqual,
  getIndexNames,
  getTensorNames,
  printExpression,
  printProgram,
  printStatement,
  walkAST,
} from './ast'

// Parser
export { parse, parseEquation, ParseError, Parser, parseRule } from './parser'

// Tokenizer
export { tokenize, Tokenizer, TokenType } from './tokenizer'

export type { Token } from './tokenizer'

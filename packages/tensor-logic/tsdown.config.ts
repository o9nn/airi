import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: {
    './index': 'src/index.ts',
    './core/index': 'src/core/index.ts',
    './parser/index': 'src/parser/index.ts',
    './inference/index': 'src/inference/index.ts',
    './autodiff/index': 'src/autodiff/index.ts',
    './embeddings/index': 'src/embeddings/index.ts',
  },
  dts: true,
  sourcemap: true,
  unused: true,
})

import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: [
    './src/index.ts',
    './src/core/index.ts',
    './src/parser/index.ts',
    './src/inference/index.ts',
    './src/autodiff/index.ts',
    './src/embeddings/index.ts',
  ],
  sourcemap: true,
  unused: true,
})

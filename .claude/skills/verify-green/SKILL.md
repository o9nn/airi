---
name: verify-green
description: Audit code whose tests and build already pass, to find the defects those signals cannot catch — bundlers that never typecheck, declared exports that were never built, tests that cannot fail, and stubs returning plausible-shaped garbage. Use when asked to "check" or "verify" work already reporting green, before claiming a package works, when inheriting a passing suite you did not write, or before reporting a task complete on code you just wrote.
---

# Verifying that green means working

A passing build and a passing suite are evidence only if they are *capable* of failing. Often they are not: the bundler never typechecked, the export map names files nobody built, and half the assertions would pass against random numbers. This skill is the audit that separates "reports green" from "works."

Run it before claiming a package works — including on your own fresh code, where the temptation to read a green suite as confirmation is strongest.

## Order of yield

Work down this list. Each step is cheap and the early ones usually explain the late ones.

### 1. Run the real typechecker — `build` is not a typecheck

`tsdown`, `tsup`, `rolldown`, `esbuild`, `swc`, and Vite all **strip** types rather than check them. A green `build` says the code parsed, nothing more.

```bash
npx tsc --noEmit 2>&1 | grep -cE 'error'
```

If that number is large, categorise before fixing anything — one bad type declaration commonly accounts for most of the list:

```bash
npx tsc --noEmit 2>&1 | grep -oE 'error TS[0-9]+' | sort | uniq -c | sort -rn
npx tsc --noEmit 2>&1 | grep -A2 'TS2345' \
  | grep -oE "Type '[^']*' is not assignable to type '[^']*'" | sort | uniq -c | sort -rn
```

Chase the dominant pair to its declaration. Fixing one default type parameter can clear a hundred errors; fixing a hundred call sites individually is the wrong move and usually means you missed the root cause.

### 2. Make every declared export resolve *and* import

`package.json` can advertise entry points the build never emits. Existence on disk is necessary but not sufficient — import each one.

```bash
node -e "const p=require('./package.json');for(const[k,v]of Object.entries(p.exports||{}))for(const[c,f]of Object.entries(v))console.log((require('fs').existsSync(f.replace(/^\.\//,''))?'OK  ':'MISS')+' '+k+' ['+c+'] '+f)"
```

```bash
for p in core parser utils; do
  node -e "import('./dist/$p/index.js').then(m=>console.log('OK   ./$p ('+Object.keys(m).length+')')).catch(e=>{console.log('FAIL ./$p '+e.message);process.exit(1)})" || break
done
```

A single-entry bundler config plus a multi-entry export map means every subpath import fails for consumers while every local test passes.

### 3. Find the tests that cannot fail

These assertions pass against garbage:

```bash
grep -rn "toBeDefined()\|toBeTruthy()\|toHaveLength\|not\.toThrow()\|isFinite\|toBeGreaterThanOrEqual(0)" --include='*.test.*' --include='*.spec.*' . | head -40
```

Stronger signal — an expectation written as prose instead of an assertion, which names the bug the author already suspected:

```bash
grep -rn "// Should\|// Would\|// For now\|// TODO\|would happen\|just verify" --include='*.test.*' --include='*.spec.*' .
```

Three shapes to recognise:

- **The commented expectation.** `expect(x).toBeDefined()` under `// Should derive 0->2`. The real claim is in the comment and is never checked.
- **The abandoned test.** `// Training would happen here...` followed by an assertion about the setup. The test name promises behaviour the body never exercises.
- **The self-test.** The body reimplements the algorithm in plain code and asserts on its own arithmetic, never calling the library. It proves the language works.

Also check the name against the body. `should reduce loss over iterations` that never iterates is a lie the suite is telling you.

### 4. Find stubs returning plausible shapes

The dangerous implementation is not the one that throws; it is the one that returns something correctly shaped and wrong.

```bash
grep -rn "For now\|Simplified\|simplified\|placeholder\|In a full\|would use\|proper implementation" --include='*.ts' --include='*.js' --include='*.py' . | grep -v test
```

`// For now, return the tensor directly` and `// Simplified: full implementation would use proper ALS` each sat above a function whose callers believed it worked.

### 5. Treat unused locals as abandoned implementations

Enable `noUnusedLocals` (or read the linter's `no-unused-vars`) and read every hit in source. A *computed-then-discarded* value is the fingerprint of a half-written algorithm:

```js
const unfolding = modeUnfold(tensor, mode)      // computed
const krProduct = khatriRaoProduct(factors, mode) // computed
return orthogonalize(randn(factors[mode].shape)) // ...and both discarded
```

That function ignored its input entirely and returned random numbers. The lint warning was the only evidence, and it reads like a nit.

## Replacing an assertion that cannot fail

Pick inputs whose answer you can derive independently of the code under test:

- **Round-trips that must be exact.** Decompose then reconstruct at full rank; encode then decode. Error must be ~0, not "finite."
- **Closed forms.** Identity matrices, one-hot vectors, known constants. Compute the expected value by hand and put the number in the test.
- **Invariants.** Distribution rows sum to 1; gradient descent on a convex objective never increases loss; orthonormal columns satisfy UᵀU = I; a boolean relation stays 0/1.
- **Ordering.** Higher rank reconstructs no worse than lower rank. Cheap to assert, hard to satisfy by accident.

Show the arithmetic in a comment so the next reader can check it without rerunning anything:

```js
// W·X = 0.6, so the sigmoid must land at 0.6457.
expect(prediction).toBeCloseTo(0.6456563, 6)
```

When an old test fails after you fix a bug, read it before editing. It may have encoded the bug — an assertion that a reconstruction has the *core's* length rather than the original's is the bug, written down as an expectation. Fix the assertion; do not relax the new one to match.

## The gate

Nothing is verified until all of these pass together, since each catches what the others cannot:

```bash
npx tsc --noEmit && npx vitest run && npm run lint && npm run build
```

Then re-import the built artifacts (step 2) — a pre-commit `lint --fix` hook can change files after your last green run.

## Reporting

Say which signals you actually ran. "Tests pass" and "typecheck, lint, tests, build, and every export path pass" are different claims, and the first one is what a green bundler tempts you into.

If you previously reported green on a weaker basis, correct it plainly and move on — one sentence, no ceremony. State test-count changes honestly: replacing four vacuous tests with four real ones is the substance, and the total barely moves.

Finish by naming what you did **not** verify. Untouched lint warnings, syntax the parser accepts but the evaluator ignores, paths with no coverage. An audit that reports only findings implies the rest is clean.

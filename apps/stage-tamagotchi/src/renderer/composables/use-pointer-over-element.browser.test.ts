import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render } from 'vitest-browser-vue'
import { computed, defineComponent, nextTick, ref } from 'vue'

import { usePointerOverElement } from './use-pointer-over-element'

describe('usePointerOverElement', () => {
  afterEach(() => {
    cleanup()
  })

  // https://github.com/moeru-ai/airi/pull/2573
  it('re-runs the canvas hit test when a DOM overlay appears under a still cursor', async () => {
    // ROOT CAUSE:
    //
    // A computed hit test only read cursor coordinates and the canvas ref.
    // Vue did not re-run elementFromPoint when a v-if overlay appeared under a still cursor.
    // We fixed this by re-running the hit test after document mutations.

    const canvas = ref<HTMLElement>()
    const x = ref(0)
    const y = ref(0)
    const showOverlay = ref(false)
    const TestHost = defineComponent({
      setup() {
        const isOver = usePointerOverElement(canvas, x, y)
        const staleOver = computed(() =>
          document.elementFromPoint(x.value, y.value) === canvas.value,
        )

        return { canvas, isOver, showOverlay, staleOver }
      },
      template: `
        <div
          ref="canvas"
          data-testid="stage-canvas"
          style="position:fixed;left:8px;top:8px;z-index:0;width:100px;height:100px;background:#4c1d95;"
        />
        <button
          v-if="showOverlay"
          data-testid="status-pill"
          style="position:fixed;left:8px;top:8px;z-index:1;width:100px;height:100px;"
        >
          status
        </button>
        <div style="position:fixed;left:200px;top:8px;">
          <span data-testid="fresh-hit">{{ isOver }}</span>
          <span data-testid="stale-hit">{{ staleOver }}</span>
        </div>
      `,
    })

    const screen = await render(TestHost)
    const canvasEl = screen.getByTestId('stage-canvas').element() as HTMLElement
    const rect = canvasEl.getBoundingClientRect()
    x.value = rect.left + 10
    y.value = rect.top + 10

    await vi.waitFor(() => {
      expect(document.elementFromPoint(x.value, y.value)).toBe(canvasEl)
      expect(screen.getByTestId('fresh-hit').element().textContent).toBe('true')
    })
    expect(screen.getByTestId('stale-hit').element().textContent).toBe('true')

    showOverlay.value = true
    await nextTick()

    await vi.waitFor(() => {
      expect(screen.getByTestId('status-pill').element()).toBeTruthy()
      expect(screen.getByTestId('fresh-hit').element().textContent).toBe('false')
    })
    expect(screen.getByTestId('stale-hit').element().textContent).toBe('true')
  })
})

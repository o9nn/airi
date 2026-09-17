import type { MaybeRefOrGetter } from 'vue'

import { useMutationObserver } from '@vueuse/core'
import { shallowRef, toValue, watch } from 'vue'

/**
 * Whether `elementFromPoint` currently resolves to `element`.
 *
 * Cursor coordinates do not change when a DOM overlay appears under a still pointer.
 * A click-through window also drops mouse events, so this re-runs after document
 * mutations as well as coordinate changes.
 */
export function usePointerOverElement(
  element: MaybeRefOrGetter<Element | null | undefined>,
  x: MaybeRefOrGetter<number>,
  y: MaybeRefOrGetter<number>,
) {
  const isOver = shallowRef(false)

  function update() {
    const target = toValue(element)
    const next = !!target && document.elementFromPoint(toValue(x), toValue(y)) === target
    if (isOver.value !== next)
      isOver.value = next
  }

  watch(
    [() => toValue(element), () => toValue(x), () => toValue(y)],
    update,
    { immediate: true, flush: 'post' },
  )

  useMutationObserver(
    () => document.documentElement,
    update,
    {
      childList: true,
      subtree: true,
      attributeFilter: ['class', 'hidden', 'style'],
    },
  )

  return isOver
}

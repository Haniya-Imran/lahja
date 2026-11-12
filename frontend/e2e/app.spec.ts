import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))

function watchErrors(page: Page) {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => { if (m.type() === 'error' && !/WebGL|GPU/i.test(m.text())) errors.push(m.text()) })
  return errors
}

test('landing renders 3D hero, live demo works, links to workspace', async ({ page }) => {
  const errors = watchErrors(page)
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('tone')
  await expect(page.locator('.fc-stamp').first()).toBeVisible({ timeout: 30000 })
  await page.getByRole('button', { name: 'Analyse', exact: true }).click()
  await expect(page.locator('.l-demo-out .badge').first()).toBeVisible({ timeout: 30000 })
  await expect(page.locator('.l-proof')).toBeVisible()
  await page.getByRole('link', { name: 'Open workspace' }).click()
  await expect(page).toHaveURL(/\/app$/)
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
  expect(errors).toEqual([])
})

test('single review: Urdu, Roman Urdu, explanation, history, validation', async ({ page }) => {
  const errors = watchErrors(page)
  await page.goto('/app/analyze')
  await page.getByRole('button', { name: 'Analyse review' }).click()
  await expect(page.getByText('Type or paste a review first.')).toBeVisible()

  const box = page.getByLabel('Review text')
  await box.fill('یہ فون بہت اچھا ہے، بیٹری بھی زبردست ہے')
  await page.getByRole('button', { name: 'Analyse review' }).click()
  await expect(page.locator('.verdict')).toBeVisible({ timeout: 30000 })
  await expect(page.locator('.badge.b-saf').first()).toHaveText('Urdu')
  await expect(page.locator('mark.tok').first()).toBeVisible()

  await page.locator('.example').filter({ hasText: /Roman Urdu/ }).first().click()
  await page.getByRole('button', { name: 'Analyse review' }).click()
  await expect(page.locator('.badge.b-saf').first()).toContainText(/Roman Urdu/, { timeout: 30000 })
  await expect(page.locator('.tbl tbody tr')).toHaveCount(2)
  expect(errors).toEqual([])
})

test('bulk CSV: upload, column suggestion, batches, filter, export, analytics', async ({ page }) => {
  const errors = watchErrors(page)
  await page.goto('/app/bulk')
  await page.setInputFiles('input[type=file]', path.join(here, '..', 'public', 'sample-reviews.csv'))
  await expect(page.getByRole('combobox').first()).toHaveValue('review', { timeout: 30000 })
  await page.getByRole('button', { name: 'Analyse all' }).click()
  await expect(page.locator('.kpi', { hasText: 'Reviews analysed' })).toContainText('12', { timeout: 60000 })
  await page.locator('select').nth(1).selectOption('positive')
  const shown = await page.locator('.tbl tbody tr').count()
  expect(shown).toBeGreaterThan(0)
  const dl = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download CSV' }).click()
  expect((await dl).suggestedFilename()).toContain('.csv')

  await page.getByRole('link', { name: 'Open analytics' }).click()
  await expect(page.locator('.kpi', { hasText: 'Net sentiment' })).toBeVisible({ timeout: 30000 })
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 30000 })
  expect(errors).toEqual([])
})

test('model, language and reports pages load real metrics', async ({ page }) => {
  const errors = watchErrors(page)
  await page.goto('/app/model')
  await expect(page.locator('.kpi', { hasText: 'Macro-F1' })).toBeVisible({ timeout: 30000 })
  await expect(page.getByText('Confusion matrix')).toBeVisible()
  await expect(page.getByText('Why this base model')).toBeVisible()
  await page.goto('/app/languages')
  await expect(page.getByText('Data sources')).toBeVisible({ timeout: 30000 })
  await page.goto('/app/reports')
  await expect(page.getByText('Nothing to export yet')).toBeVisible()
  expect(errors).toEqual([])
})

test('mobile layout has no horizontal scroll', async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } })
  const page = await ctx.newPage()
  for (const url of ['/', '/app', '/app/analyze', '/app/bulk']) {
    await page.goto(url)
    await page.waitForLoadState('networkidle')
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(over, url).toBeLessThanOrEqual(1)
  }
  await ctx.close()
})

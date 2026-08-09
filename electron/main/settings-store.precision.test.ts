import assert from 'node:assert/strict'
import { mkdtemp, rm, readFile, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'

async function loadModule() {
  return import(new URL('./settings-store.ts', import.meta.url).href)
}

async function withTempDir(run: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(path.join(tmpdir(), 'modly-settings-'))
  try {
    await run(dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

test('Precision type includes all expected values via runtime checks', async () => {
  const { getSettings, setSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const validPrecisions = ['auto', 'fp16', 'bf16', 'fp32'] as const
    for (const precision of validPrecisions) {
      const updated = setSettings(dir, { precision })
      assert.equal(updated.precision, precision)
      const read = getSettings(dir)
      assert.equal(read.precision, precision)
    }
  })
})

test('ComputeDevice type includes all expected values via runtime checks', async () => {
  const { setSettings, getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const validDevices = ['auto', 'mps', 'cuda0', 'cuda1', 'cpu'] as const
    for (const device of validDevices) {
      const updated = setSettings(dir, { device })
      assert.equal(updated.device, device)
      const read = getSettings(dir)
      assert.equal(read.device, device)
    }
  })
})

test('default settings include device: "auto" and precision: "auto"', async () => {
  const { getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const settings = getSettings(dir)
    assert.equal(settings.device, 'auto')
    assert.equal(settings.precision, 'auto')
  })
})

test('getSettings returns an object with all expected directory fields', async () => {
  const { getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const settings = getSettings(dir)
    assert.equal(settings.modelsDir, path.join(dir, 'models'))
    assert.equal(settings.workspaceDir, path.join(dir, 'workspace'))
    assert.equal(settings.workflowsDir, path.join(dir, 'workflows'))
    assert.equal(settings.extensionsDir, path.join(dir, 'extensions'))
    assert.equal(settings.dependenciesDir, path.join(dir, 'dependencies'))
  })
})

test('getSettings returns defaults when no settings file exists', async () => {
  const { getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    assert.ok(!existsSync(path.join(dir, 'settings.json')))
    const settings = getSettings(dir)
    assert.equal(settings.device, 'auto')
    assert.equal(settings.precision, 'auto')
  })
})

test('setSettings writes a settings.json file', async () => {
  const { setSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { device: 'mps', precision: 'bf16' })
    assert.ok(existsSync(path.join(dir, 'settings.json')))
  })
})

test('setSettings persists device and precision to disk', async () => {
  const { setSettings, getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { device: 'mps', precision: 'bf16' })
    const settings = getSettings(dir)
    assert.equal(settings.device, 'mps')
    assert.equal(settings.precision, 'bf16')
  })
})

test('setSettings merges patch over existing settings', async () => {
  const { setSettings, getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { device: 'mps', precision: 'bf16' })
    setSettings(dir, { precision: 'fp32' })
    const settings = getSettings(dir)
    assert.equal(settings.device, 'mps')
    assert.equal(settings.precision, 'fp32')
  })
})

test('setSettings returns the fully merged settings object', async () => {
  const { setSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const result = setSettings(dir, { device: 'cpu', precision: 'fp16' })
    assert.equal(result.device, 'cpu')
    assert.equal(result.precision, 'fp16')
    assert.equal(result.modelsDir, path.join(dir, 'models'))
  })
})

test('getSettings migrates legacy outputsDir key to workspaceDir', async () => {
  const { getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    const legacy = {
      modelsDir: path.join(dir, 'models'),
      outputsDir: path.join(dir, 'legacy-outputs'),
    }
    await writeFile(path.join(dir, 'settings.json'), JSON.stringify(legacy))
    const settings = getSettings(dir)
    assert.equal(settings.workspaceDir, path.join(dir, 'legacy-outputs'))
    assert.equal(settings.outputsDir, undefined)
  })
})

test('getSettings returns defaults when settings.json is invalid JSON', async () => {
  const { getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    await writeFile(path.join(dir, 'settings.json'), '{ not valid json')
    const settings = getSettings(dir)
    assert.equal(settings.device, 'auto')
    assert.equal(settings.precision, 'auto')
  })
})

test('setSettings preserves other fields when updating only device', async () => {
  const { setSettings, getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { precision: 'bf16', hfToken: 'test-token' })
    setSettings(dir, { device: 'cuda0' })
    const settings = getSettings(dir)
    assert.equal(settings.device, 'cuda0')
    assert.equal(settings.precision, 'bf16')
    assert.equal(settings.hfToken, 'test-token')
  })
})

test('setSettings can clear device and precision back to auto', async () => {
  const { setSettings, getSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { device: 'mps', precision: 'bf16' })
    setSettings(dir, { device: 'auto', precision: 'auto' })
    const settings = getSettings(dir)
    assert.equal(settings.device, 'auto')
    assert.equal(settings.precision, 'auto')
  })
})

test('settings.json on disk contains device and precision keys', async () => {
  const { setSettings } = await loadModule()
  await withTempDir(async (dir) => {
    setSettings(dir, { device: 'mps', precision: 'bf16' })
    const raw = await readFile(path.join(dir, 'settings.json'), 'utf-8')
    const parsed = JSON.parse(raw)
    assert.equal(parsed.device, 'mps')
    assert.equal(parsed.precision, 'bf16')
  })
})
